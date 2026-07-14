# 第 29 章 SGLang Scheduler 设计

上一章讲了 RadixAttention：SGLang Runtime 用 radix tree 管理 token prefix 到 KV cache 的映射，通过最长前缀匹配复用已有 KV，只对未命中 suffix 做 prefill。

但有 cache 还不够。

Runtime 每一轮都要决定：哪些请求继续 decode，哪些新请求可以 prefill，哪些请求因为 KV cache 不够要等待，哪些 cache 命中请求应该优先，哪些长 prompt 要切分，哪些 structured output 请求会带来额外开销。

这些决策由 scheduler 完成。

一句话概括：

> SGLang scheduler 是 runtime 的资源分配中枢，它在每个 engine step 结合 waiting queue、running requests、RadixAttention 命中、KV cache 容量、token budget、structured output 状态和公平性策略，决定本轮执行哪些 prefill 和 decode 工作。

## 29.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求做过资料校准，主要参考五类公开资料：

1. SGLang 论文《SGLang: Efficient Execution of Structured Language Model Programs》对 frontend language、runtime、RadixAttention、KV cache reuse 和 structured decoding 的系统分层说明。
2. SGLang Server Arguments 文档对 `--mem-fraction-static`、`--max-running-requests`、`--max-total-tokens`、`--chunked-prefill-size`、`--max-prefill-tokens`、`--schedule-policy`、`--schedule-conservativeness`、`--disable-radix-cache` 和 `--enable-metrics` 等 serving / scheduler 参数的公开口径。
3. SGLang Hyperparameter Tuning 文档对 `#queue-req`、`token usage`、KV cache pool full / retract、`--schedule-conservativeness`、`--mem-fraction-static`、`--chunked-prefill-size`、`--max-running-requests` 和 `--schedule-policy lpm` 的调优建议。
4. SGLang Structured Outputs 文档对 JSON schema、regex、EBNF、structural tag、grammar backend 和约束输出接口的说明。
5. SGLang Production Metrics / Attention Backend 文档对 prompt tokens、generation tokens、token usage、cache hit rate、TTFT、E2E latency、TPOT、running requests、queue requests，以及 page size / prefix cache 完整页命中的说明。

本章只讲 SGLang-like scheduler 的教学版机制：waiting / running 状态、decode-first、cache-aware admission、prefill / decode token budget、sequence budget、KV budget、chunked prefill、structured output 开销、abort cleanup、eviction 与公平性。它不绑定某个 SGLang 版本的真实源码类名、真实 schedule policy 实现、CUDA graph、overlap scheduler、DP attention、PD disaggregation、LoRA adapter 调度、MoE 路由或生产参数全集。本章 demo 用纯 Python 模拟调度决策和 KV slot 账本，不执行真实模型 forward。

参考资料：

1. SGLang 论文：<https://arxiv.org/abs/2312.07104>
2. SGLang Server Arguments：<https://docs.sglang.io/docs/advanced_features/server_arguments.md>
3. SGLang Hyperparameter Tuning：<https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning.md>
4. SGLang Structured Outputs：<https://docs.sglang.io/docs/advanced_features/structured_outputs.md>
5. SGLang Production Metrics：<https://docs.sglang.io/docs/references/production_metrics.md>
6. SGLang Attention Backend：<https://docs.sglang.io/docs/advanced_features/attention_backend.md>

## 29.1 本章目标

读完本章，你应该能讲清：

1. SGLang scheduler 和普通 continuous batching scheduler 的共同点。
2. SGLang scheduler 为什么必须理解 RadixAttention 命中长度。
3. waiting、running、prefill、decode、finished 等状态如何变化。
4. token budget、sequence budget、KV budget 分别约束什么。
5. cache-aware scheduling 如何影响 TTFT、吞吐和公平性。
6. structured output、streaming、abort、frontend program 分支为什么会影响调度。
7. 面试中如何画出 SGLang scheduler 的核心流程。

## 29.2 先看 scheduler 在 runtime 中的位置

SGLang Runtime 的主链路可以简化成：

```text
API / offline engine / frontend program
  -> request parsing
  -> tokenization
  -> request state
  -> scheduler
  -> RadixAttention / memory pool
  -> model runner
  -> sampler / grammar backend
  -> output / streaming
```

Scheduler 位于 request state 和 model execution 之间。

它不直接执行 GPU kernel，也不直接采样 token。

它负责产生本轮执行计划：

```text
本轮哪些请求做 prefill？
本轮哪些请求做 decode？
每个请求处理多少 token？
需要分配哪些 KV cache？
复用哪些 cached prefix？
哪些请求要等待、暂停、结束或清理？
```

如果说 model runner 是“执行器”，memory pool 是“显存账本”，RadixAttention 是“prefix cache 索引”，那么 scheduler 就是“调度决策者”。

## 29.3 Scheduler 的输入

每个 engine step，scheduler 通常需要这些输入：

1. 新到达的 requests。
2. waiting queue。
3. running requests。
4. finished 或 aborted requests。
5. 每个请求的 token ids。
6. 每个请求已计算 token 数。
7. 每个请求的 output token 数。
8. sampling parameters。
9. structured output / grammar state。
10. RadixAttention 当前树状态。
11. 每个 waiting 请求的 cache hit length。
12. KV memory pool 的 free slots。
13. token budget。
14. sequence budget。
15. 调度策略配置。

用伪代码表示：

```python
class SchedulerInput:
    waiting: list[Request]
    running: list[Request]
    radix_cache: RadixCache
    memory_pool: KVMemoryPool
    max_batched_tokens: int
    max_running_requests: int
    policy: SchedulingPolicy
```

这些输入说明 scheduler 不是只看“请求列表”。

它要同时看计算预算、显存预算、cache 命中、请求状态和输出约束。

## 29.4 Scheduler 的输出

Scheduler 的输出是本轮执行计划。

一个简化输出可以是：

```python
class SchedulerOutput:
    prefill_batch: list[PrefillWork]
    decode_batch: list[DecodeWork]
    cache_hits: dict[str, CacheHitInfo]
    kv_allocations: dict[str, list[KVSlot]]
    evictions: list[CacheNode]
    aborted: list[str]
```

更直观地说：

```text
本轮执行计划
  -> prefill 哪些 tokens
  -> decode 哪些 requests
  -> 读取哪些 cached KV
  -> 写入哪些 KV slots
  -> 哪些请求本轮不执行
  -> 哪些资源要释放或淘汰
```

这个输出会同时交给：

1. memory pool，用于分配、追加和释放 KV cache。
2. model runner，用于准备 input ids、positions、slot mapping 和 attention metadata。
3. sampler，用于知道哪些 logits 需要采样。
4. output layer，用于更新请求和流式返回。

所以 scheduler 输出不是简单的 request ids，而是 execution metadata 的源头。

## 29.5 请求状态机

SGLang Runtime 中，一个请求可以用下面的状态机理解：

```text
WAITING
  -> PREFILLING
  -> DECODING
  -> FINISHED

WAITING
  -> PREFILLING
  -> WAITING / PREEMPTED
  -> DECODING
  -> ABORTED / FAILED
```

每个状态含义：

WAITING：请求已进入 runtime，但还没有完成 prefill。

PREFILLING：本轮正在处理 prompt 或未命中 suffix。

DECODING：已经有历史 KV，每轮生成新 token。

FINISHED：遇到 EOS、stop、max_new_tokens、grammar 完成或其他正常结束条件。

ABORTED：客户端断开、用户取消、上层超时或系统主动中止。

FAILED：内部错误、KV 分配失败、模型执行错误等。

和普通 serving 相比，SGLang 的请求状态还可能关联：

1. RadixAttention 命中路径。
2. 当前 prefix cache 引用。
3. frontend program 中的变量或分支。
4. grammar state。
5. 多个并行 generation call。

所以“一个请求”在 SGLang 里经常不仅是一段 prompt，而是复杂程序执行过程中的一个 generation 节点。

## 29.6 Engine step：调度的基本节拍

Scheduler 不是只在请求到来时运行一次，而是在每个 engine step 运行。

一个简化主循环：

```python
while True:
    new_requests = receive_new_requests()
    waiting_queue.extend(new_requests)

    cleanup_finished_and_aborted()

    schedule = scheduler.schedule(
        waiting=waiting_queue,
        running=running_requests,
        radix_cache=radix_cache,
        memory_pool=memory_pool,
    )

    outputs = model_runner.forward(schedule)
    next_tokens = sampler.sample(outputs)
    update_request_states(next_tokens)
    stream_outputs(next_tokens)
```

这个循环就是 serving engine 的心跳。

每一轮都可能发生：

1. 新请求进入 waiting。
2. 老请求继续 decode。
3. 短请求结束并释放资源。
4. 长请求追加 KV。
5. cache 命中请求进入 prefill。
6. cache miss 请求继续等待。
7. structured output 请求更新 grammar state。

## 29.7 Prefill 和 decode 为什么要一起调度

Prefill 和 decode 的资源特点不同。

Prefill：

1. 一次处理多个 prompt token。
2. 影响 TTFT。
3. 计算量大。
4. 写入大量 KV cache。
5. 长 prompt 可能占用较长 GPU 时间。

Decode：

1. 每个请求每轮通常一个 token。
2. 影响 TPOT 和 streaming 平滑度。
3. 更依赖 KV cache 读取。
4. batch 动态变化。
5. 会逐步追加 KV cache。

如果 scheduler 过度偏向 prefill：

```text
新请求 TTFT 变好
正在 streaming 的请求 TPOT 抖动
用户看到输出卡顿
```

如果 scheduler 过度偏向 decode：

```text
已有请求输出平滑
waiting queue 堆积
新请求 TTFT 变差
```

因此 scheduler 的核心工作就是在 TTFT、TPOT、吞吐和显存之间做取舍。

## 29.8 最小调度策略

一个最小策略可以这样写：

```python
def schedule_step(waiting, running, max_tokens, max_seqs):
    scheduled_decode = []
    scheduled_prefill = []
    used_tokens = 0
    used_seqs = 0

    for req in running:
        if used_tokens + 1 > max_tokens:
            break
        if used_seqs + 1 > max_seqs:
            break
        scheduled_decode.append(req)
        used_tokens += 1
        used_seqs += 1

    while waiting:
        req = waiting.peek()
        prefill_tokens = len(req.input_ids)
        if used_tokens + prefill_tokens > max_tokens:
            break
        if used_seqs + 1 > max_seqs:
            break
        scheduled_prefill.append(waiting.pop())
        used_tokens += prefill_tokens
        used_seqs += 1

    return scheduled_prefill, scheduled_decode
```

这只是教学版。

它忽略了：

1. RadixAttention 命中。
2. KV cache 容量。
3. chunked prefill。
4. structured output 开销。
5. 请求优先级。
6. cache eviction。
7. abort 和 cleanup。
8. frontend program 分支。

真实 SGLang scheduler 的关键就是把这些因素加进去。

## 29.9 Token budget：本轮最多算多少 token

Token budget 限制本轮模型 forward 的 token 工作量。

可以理解成：

```text
scheduled_tokens = prefill_tokens + decode_tokens
scheduled_tokens <= max_batched_tokens
```

Decode tokens 通常近似等于本轮 decode 请求数：

```text
decode_tokens = number_of_running_requests
```

Prefill tokens 取决于新请求需要计算的 prompt tokens。

在 SGLang 中，重点是：prefill tokens 应该按未命中 suffix 计算，而不是总 prompt 长度。

例如：

```text
prompt length = 12000
cache hit length = 10000
uncached suffix = 2000
```

对本轮 prefill 计算来说，更相关的是 2000，而不是 12000。

这就是 RadixAttention 和 scheduler 的交点。

## 29.10 Sequence budget：本轮最多跑多少序列

Sequence budget 限制本轮 batch 中的请求数量。

即使 token budget 还够，也不能无限增加 sequence 数。

原因包括：

1. 每个 sequence 都有 metadata 开销。
2. attention backend 对 batch size 有上限或性能拐点。
3. sampler 要处理每个 sequence 的 logits。
4. structured output 每个 sequence 可能有 grammar state。
5. streaming 输出也有 per-request 开销。

可以写成：

```text
scheduled_sequences <= max_num_seqs
```

Sequence budget 控制的是“请求数维度”的复杂度，token budget 控制的是“token 计算量维度”的复杂度。

两者都必须有。

## 29.11 KV budget：能算不代表放得下

Token budget 和 sequence budget 通过了，不代表请求能执行。

还要看 KV cache 能不能放下。

每个新 prefill token 都会写 KV cache。

每个 decode token 也会追加 KV cache。

因此 scheduler 每轮要检查：

```text
required_new_kv_slots <= free_kv_slots + evictable_cached_slots
```

如果 KV cache 不够，有几种选择：

1. 不接纳新的 prefill 请求。
2. 减少本轮 prefill tokens。
3. 做 cache eviction。
4. 抢占或暂停部分请求。
5. 返回过载错误或排队等待。

SGLang 因为有 RadixAttention，还要注意：有些 KV cache 虽然没有 active request 引用，但作为 prefix cache 保留在 radix tree 中。

这些 cache 是否可以淘汰，要由 memory pool 和 RadixAttention 协同决定。

## 29.12 Cache hit length 如何进入调度

对 waiting request，scheduler 可以先询问 RadixAttention：

```text
这个请求能命中多少 token？
命中的 KV cache 是否仍在 memory pool 中？
命中的节点是否可以 pin 住？
未命中 suffix 需要多少新 KV slots？
```

得到结果：

```python
class CacheHitInfo:
    hit_len: int
    suffix_len: int
    matched_nodes: list[RadixNode]
    kv_refs: list[KVRef]
```

然后 scheduler 用 `suffix_len` 估算 prefill cost。

伪代码：

```python
hit = radix_cache.match(req.input_ids)
prefill_cost = len(req.input_ids) - hit.hit_len
kv_cost = prefill_cost + req.expected_decode_reserve

if can_fit(prefill_cost, kv_cost):
    pin(hit.matched_nodes)
    schedule_prefill(req, hit)
```

这里的 `pin` 很关键。

一旦 scheduler 决定复用某段 prefix，就必须防止这段 KV 在本轮执行前被淘汰。

## 29.13 Cache-aware admission

Admission 指 waiting 请求能不能进入本轮执行。

普通 admission 可能看：

```text
prompt_len <= remaining_token_budget
required_kv <= free_kv
```

Cache-aware admission 应该看：

```text
uncached_suffix_len <= remaining_token_budget
new_kv_required <= free_kv_after_possible_eviction
```

例如：

```text
R1: prompt 10000, hit 9000, suffix 1000
R2: prompt 3000, hit 0, suffix 3000
```

如果只看 prompt 长度，R2 更容易进入。

如果看实际 prefill cost，R1 更便宜。

Cache-aware admission 可以提升吞吐和 TTFT，但要小心公平性。如果一直优先 cache hit 请求，cache miss 请求可能长期等待。

## 29.14 Decode-first 策略

很多 serving scheduler 会倾向于先保证 running decode。

原因是：

1. Running 请求已经占用了 KV cache。
2. Decode 直接影响 streaming 用户体验。
3. 每轮 decode 不推进，TPOT 就会上升。
4. Decode 请求通常每轮只需要一个 token，比较容易塞进 budget。

Decode-first 的直觉策略：

```text
先为 running requests 安排 decode
再用剩余 budget 安排 waiting prefill
```

优点：

1. Streaming 更平滑。
2. 已进入系统的请求更快完成。
3. KV cache 占用更快释放。

缺点：

1. 高负载时 waiting queue 可能堆积。
2. 新请求 TTFT 可能变差。
3. 长时间 decode-heavy 时 prefill 饥饿。

所以 decode-first 通常还需要配合 prefill 配额或 aging 策略。

## 29.15 Prefill 配额和防饥饿

为了避免 waiting 请求长期得不到 prefill，scheduler 可以给 prefill 留预算。

例如：

```text
每轮最多 80% token budget 给 decode
至少保留 20% 给 prefill
```

或者：

```text
waiting 请求等待超过阈值后提升优先级
```

这类策略解决的是公平性。

简单 aging 伪代码：

```python
def priority(req, now):
    wait_time = now - req.arrival_time
    cache_bonus = req.cache_hit_len * 0.001
    return wait_time + cache_bonus
```

这里 cache hit 可以加分，但 wait time 也会持续增长，避免 miss 请求永远被饿死。

真实系统策略会更复杂，但思想类似：不能只追求局部吞吐最大化。

## 29.16 Chunked prefill

长 prompt prefill 可能阻塞 decode。

例如：

```text
一个 64000-token prompt 进入系统
```

如果一次性 prefill，GPU 可能长时间被这个请求占用，其他 streaming 请求 TPOT 抖动。

Chunked prefill 的思想是把长 prompt 拆成多个 chunk：

```text
chunk 1: tokens 0-4095
chunk 2: tokens 4096-8191
chunk 3: tokens 8192-12287
...
```

每个 engine step 只处理一部分 prefill，让 decode 有机会穿插执行。

SGLang 中如果有 RadixAttention 命中，chunked prefill 应该作用在未命中 suffix 上：

```text
prompt length = 64000
cache hit = 48000
suffix = 16000

chunked prefill 只处理 suffix 的 16000 tokens
```

Chunked prefill 的收益：

1. 降低长 prompt 对 TPOT 的冲击。
2. 改善 p95/p99 streaming 抖动。
3. 让 scheduler 更细粒度地混合 prefill 和 decode。

代价：

1. 请求完成 prefill 需要多个 step。
2. 状态管理更复杂。
3. 可能增加调度和 metadata 开销。

## 29.17 Structured output 对调度的影响

Structured output 看起来是 sampler 或 grammar backend 的事，但它也会影响 scheduler。

原因包括：

1. 每个请求有独立 grammar state。
2. 每轮 decode 需要计算合法 token mask。
3. 复杂 schema 可能增加 CPU/GPU 开销。
4. 某些约束会改变结束条件。
5. Tool call 或 structural tag 可能影响后续 frontend program 分支。

例如 JSON schema 请求可能在生成完整 JSON 后结束，而不是只看 EOS。

Scheduler 至少要知道：

1. 这个请求是否启用 grammar constraint。
2. 它是否可能带来额外 per-step 开销。
3. 它是否已经被 grammar state 判定完成。
4. 它的 max_new_tokens 是否只是上限，不是实际结束时刻。

如果一个 batch 中结构化请求很多，decode step latency 可能上升。

## 29.18 Frontend program 分支对调度的影响

SGLang 的特殊之处在于，它不只服务独立 chat 请求，还可能执行 frontend language 产生的复杂程序。

例如：

```text
gen root
  -> fork 4 branches
      -> gen branch answer
  -> merge
  -> gen final answer
```

这里会产生多个 generation work items。

这些 work items 可能：

1. 共享同一个 prefix。
2. 可以并行执行。
3. 有依赖关系。
4. 某些分支完成后触发后续请求。
5. 某些分支失败后取消其他分支。

Scheduler 不一定直接理解完整业务语义，但 runtime 需要能处理这种动态生成的请求流。

这也是 SGLang scheduler 比普通单请求 server 更复杂的地方：请求不是只从 HTTP 入口来，还可能由正在执行的 LLM program 派生出来。

## 29.19 Streaming 和 abort

Streaming 请求需要稳定 decode。

如果 scheduler 长时间不调度某个 running 请求，用户看到的输出会卡顿。

同时，streaming 客户端可能断开。

断开后，runtime 必须 abort 请求：

```text
client disconnect
  -> mark request aborted
  -> remove from running/waiting
  -> release active KV refs
  -> decide whether generated prefix cache can remain
```

注意最后一点：请求 abort 不一定意味着所有 KV 都必须立即丢弃。

如果已经生成的前缀有复用价值，并且隔离策略允许，runtime 可以保留部分 cache。

但 active request 引用必须清理，否则会造成显存泄漏。

## 29.20 Eviction 和调度的交互

当 KV budget 不够时，scheduler 可能触发 eviction。

Eviction 不是简单删缓存，它要满足：

1. 不能淘汰 active request 正在用的 KV。
2. 尽量淘汰 ref count 为 0 的 radix tree 叶子。
3. 淘汰后要更新 radix tree。
4. 淘汰后要更新 memory pool free slots。
5. 不能让已计划执行的 cache hit 失效。

一个安全顺序是：

```text
1. 估算本轮需要 KV slots
2. 找到可淘汰 cache nodes
3. evict 并释放 slots
4. pin 本轮要复用的 cached prefix
5. allocate 新 slots
6. 生成 schedule
```

如果顺序错了，可能出现刚命中的 cache 被淘汰，或者 model runner 读取无效 KV。

## 29.21 Tail latency：p99 为什么难

Scheduler 不只是追求平均吞吐，还要控制 p95/p99。

尾延迟常见来源：

1. 长 prompt 突然进入。
2. cache miss 请求排队。
3. KV cache 接近满导致频繁 eviction。
4. decode batch 过大导致 step latency 上升。
5. structured output 请求集中出现。
6. tokenizer 或前置 RAG 抖动。
7. 某些请求生成特别长。
8. abort cleanup 不及时。

Scheduler 能做的缓解包括：

1. 限制单 step prefill tokens。
2. chunked prefill。
3. 控制 running batch size。
4. aging 防饥饿。
5. 根据 cache hit 调整 admission。
6. 对超长请求做隔离队列。
7. 对高优先级流量使用独立 runtime 或队列。

尾延迟不是一个参数能解决的，它是调度、显存、流量形态和上游链路共同作用的结果。

## 29.22 一个完整调度例子

假设当前状态：

```text
running:
  R1, R2, R3  # 都在 decode

waiting:
  W1: prompt 12000, cache hit 10000, suffix 2000
  W2: prompt 3000,  cache hit 0,     suffix 3000
  W3: prompt 8000,  cache hit 7800,  suffix 200

budget:
  max tokens this step = 4096
  max seqs this step = 6
```

Decode-first 后：

```text
schedule decode: R1, R2, R3
used tokens = 3
used seqs = 3
remaining tokens = 4093
remaining seqs = 3
```

如果按 prompt length 排队，可能先看 W1，发现 12000 放不下。

如果按 suffix cost 和 cache-aware admission：

```text
W3 suffix 200  -> 放入 prefill
W1 suffix 2000 -> 放入 prefill
W2 suffix 3000 -> 剩余 tokens 不够，等待
```

本轮计划：

```text
decode: R1, R2, R3
prefill: W3 suffix 200, W1 suffix 2000
total tokens = 3 + 200 + 2000 = 2203
seqs = 5
```

这就是 cache-aware scheduling 的直觉：prompt 总长度不是唯一成本，未命中 suffix 才是本轮 prefill 的关键成本。

## 29.23 一个简化 scheduler 伪代码

下面伪代码把本章概念串起来：

```python
def schedule_step(waiting, running, radix_cache, memory_pool, config):
    budget = TokenBudget(config.max_batched_tokens, config.max_num_seqs)
    scheduled_decode = []
    scheduled_prefill = []

    cleanup_aborted_and_finished(running, waiting, memory_pool)

    for req in running:
        if not budget.can_add(tokens=1, seqs=1):
            break
        if not memory_pool.can_append_one_token(req):
            break
        scheduled_decode.append(req)
        budget.add(tokens=1, seqs=1)

    candidates = []
    for req in waiting:
        hit = radix_cache.match(req.input_ids)
        suffix_len = len(req.input_ids) - hit.hit_len
        candidates.append((req, hit, suffix_len))

    candidates = order_by_policy(candidates)

    for req, hit, suffix_len in candidates:
        chunk_len = min(suffix_len, config.max_prefill_chunk)
        if not budget.can_add(tokens=chunk_len, seqs=1):
            continue
        if not memory_pool.can_allocate(chunk_len):
            evict_for(chunk_len, radix_cache, memory_pool)
        if not memory_pool.can_allocate(chunk_len):
            continue

        radix_cache.pin(hit.nodes)
        slots = memory_pool.allocate(chunk_len)
        scheduled_prefill.append(PrefillWork(req, hit, slots, chunk_len))
        budget.add(tokens=chunk_len, seqs=1)

    return SchedulerOutput(scheduled_prefill, scheduled_decode)
```

这不是 SGLang 真实源码，只是教学用骨架。

它表达几个关键点：

1. 先清理 finished/aborted。
2. Decode 和 prefill 都要进入同一个 token budget。
3. Waiting 请求先做 RadixAttention match。
4. 调度 prefill 看 suffix length，而不是 prompt length。
5. KV 不够时需要 eviction。
6. 复用 cache 前要 pin。
7. 最终输出的是执行计划。

## 29.24 常见误解

误解一：Scheduler 只是把请求凑成 batch。

不对。Scheduler 要同时处理 prefill、decode、KV cache、RadixAttention、budget、公平性、abort 和 streaming。

误解二：Cache hit 请求一定优先。

不一定。Cache hit 能降低成本，但如果过度优先，会让 cache miss 请求饥饿。真实策略要平衡吞吐和公平性。

误解三：只要 token budget 够，请求就能执行。

不对。KV cache 也要够，sequence budget 也要够，structured output 和 metadata 开销也要考虑。

误解四：Prefill 越快越好，应该尽量多塞。

不一定。Prefill 太多会影响 running decode 的 TPOT 和 streaming 平滑度。

误解五：Structured output 只影响 sampler，不影响 scheduler。

不准确。Grammar mask、schema 状态和结构化结束条件会影响每轮 decode 成本和请求生命周期。

## 29.25 面试官会怎么问

问题一：SGLang scheduler 和 vLLM scheduler 有什么共同点？

回答要点：都要做 iteration-level scheduling、continuous batching、prefill/decode 平衡、token budget、sequence budget、KV cache 管理和 streaming 请求状态更新。

问题二：SGLang scheduler 有什么额外特点？

回答要点：它要和 RadixAttention 协同，调度时考虑 prefix cache hit length、uncached suffix cost、cache-aware admission、radix tree eviction/pin；还要适配 SGLang frontend program 的分支、多次 generation 和 structured output。

问题三：为什么 scheduler 不能只看 prompt length？

回答要点：因为 RadixAttention 命中后，真正需要 prefill 的只是未命中 suffix。一个 12000-token prompt 如果命中 10000 tokens，本轮 prefill 成本更接近 2000 tokens。

问题四：为什么 scheduler 要平衡 prefill 和 decode？

回答要点：prefill 影响新请求 TTFT，decode 影响已有请求 TPOT 和 streaming 平滑度。过度偏向任何一边都会导致另一类指标恶化。

问题五：KV cache 不够时 scheduler 怎么办？

回答要点：可以拒绝或延后新 prefill、减少 chunk、触发 RadixAttention cache eviction、暂停/抢占部分请求或返回过载。eviction 必须避开 active request 正在使用的 KV。

## 29.26 标准回答模板

如果面试官问“SGLang scheduler 怎么设计”，可以这样回答：

```text
SGLang scheduler 是 runtime 的资源分配中枢。它在每个 engine step 根据 waiting queue、running requests、token budget、sequence budget、KV cache budget、RadixAttention 命中情况和请求状态，决定本轮执行哪些 prefill 和 decode。

它和 vLLM-like continuous batching scheduler 一样，需要平衡 prefill 和 decode：prefill 影响 TTFT，decode 影响 TPOT 和 streaming 平滑度。不同的是，SGLang scheduler 还要和 RadixAttention 协同。对 waiting 请求，它不能只看 prompt length，而要先做 prefix match，得到 cache hit length 和 uncached suffix length。真正的 prefill 成本主要是 suffix length；如果决定复用 cache，还要 pin 住对应 radix tree 节点，避免执行前被淘汰。

调度时还要检查 KV cache 是否足够。KV 不够时，可以延后请求、做 chunked prefill、触发 radix tree 叶子 cache eviction，或者在极端情况下抢占/拒绝请求。对于 structured output 请求，scheduler 还要意识到 grammar state 和 token mask 会增加每步 decode 开销，并影响请求结束条件。

所以 SGLang scheduler 不只是凑 batch，而是在 TTFT、TPOT、吞吐、KV 显存、cache hit、公平性和复杂程序执行之间做权衡。
```

## 29.27 SGLang Scheduler 公式、门禁和可运行 demo

把第 `i` 个 waiting 请求的 prompt token 数记为 `L_i`，RadixAttention 命中长度记为 `H_i`。如果底层 page size 为 `S`，则 page 对齐后的可复用 prefix token 数为：

$$
H_i^{\mathrm{page}}=S\left\lfloor \frac{H_i}{S}\right\rfloor
$$

本轮 prefill 真正需要计算的 suffix token 数为：

$$
P_i^{\mathrm{run}}=\max(0,L_i-H_i^{\mathrm{page}}-C_i)
$$

其中 `C_i` 是这个请求之前已经 chunked prefill 完成的 suffix token 数。若本轮 chunk size 上限为 `B_{\mathrm{chunk}}`，则本轮最多 prefill：

$$
Q_i=\min(P_i^{\mathrm{run}},B_{\mathrm{chunk}})
$$

在第 `t` 个 engine step 中，令 `D_t` 是 scheduled decode token 数，`Q_t` 是 scheduled prefill token 数，`N_t` 是 scheduled sequence 数，则 token / sequence budget 可以写成：

$$
D_t+Q_t\le B_{\mathrm{tok}},\qquad N_t\le B_{\mathrm{seq}}
$$

KV budget 还要满足：

$$
K_t^{\mathrm{new}}\le F_t+E_t
$$

其中 `K_t^{\mathrm{new}}` 是本轮新增 KV slots，`F_t` 是当前 free slots，`E_t` 是可安全 eviction 的 cached slots。注意 `E_t` 不能包含 active request 正在引用或本轮刚 pin 住的 radix prefix。

为了避免 cache hit 请求长期压过 cache miss 请求，可以给 waiting request 加 aging 分数：

$$
A_i(t)=(t-a_i)+\lambda H_i^{\mathrm{page}}
$$

其中 `a_i` 是到达 step，`\lambda` 是 cache hit bonus。`A_i(t)` 的第一项会随等待时间增长，保证长期等待的 cache miss 请求最终也能被调度。

教学版 SGLang scheduler 门禁可以写成：

$$
G_{\mathrm{scheduler}}=G_{\mathrm{decode}}G_{\mathrm{tok}}G_{\mathrm{seq}}G_{\mathrm{suffix}}G_{\mathrm{kv}}G_{\mathrm{fair}}G_{\mathrm{chunk}}G_{\mathrm{grammar}}G_{\mathrm{cleanup}}G_{\mathrm{metric}}
$$

其中：

1. `G_{\mathrm{decode}}`：running decode 能优先推进，避免 streaming TPOT 抖动。
2. `G_{\mathrm{tok}}`：prefill + decode 不超过 token budget。
3. `G_{\mathrm{seq}}`：本轮 request / sequence 数不超过 sequence budget。
4. `G_{\mathrm{suffix}}`：waiting prefill cost 使用未命中 suffix，而不是原始 prompt length。
5. `G_{\mathrm{kv}}`：KV 不足时只淘汰安全 cached nodes，不破坏 active / pinned prefix。
6. `G_{\mathrm{fair}}`：cache hit bonus 不能让 cache miss 请求无限饥饿。
7. `G_{\mathrm{chunk}}`：长 suffix 能按 chunk 分多轮 prefill。
8. `G_{\mathrm{grammar}}`：structured output 请求的 per-step grammar 开销可见。
9. `G_{\mathrm{cleanup}}`：abort / finished 请求能释放 active KV refs。
10. `G_{\mathrm{metric}}`：能输出 scheduled tokens、queue / deferred reason、KV free、eviction、grammar steps 和 gate。

下面这个 0 依赖 demo 模拟一个 engine step：两个 streaming 请求先 decode，一个已取消请求释放 KV；waiting queue 中既有 cache hit 长 prompt，也有等待很久的 cache miss 请求。Scheduler 用 aging + cache bonus 排序，用 suffix cost 而不是 prompt length 计入 budget，在 KV 不够时淘汰 stale cached nodes，并把 matched radix prefix pin 住。

```python
from dataclasses import dataclass


@dataclass
class Request:
    req_id: str
    prompt_len: int
    hit_len: int
    arrival_step: int
    max_new_tokens: int
    phase: str = "WAITING"
    active_slots: int = 0
    generated: int = 0
    grammar: bool = False
    stream: bool = False
    aborted: bool = False
    computed_suffix: int = 0

    @property
    def suffix_len(self):
        return max(0, self.prompt_len - self.hit_len)

    @property
    def remaining_suffix(self):
        return max(0, self.suffix_len - self.computed_suffix)


@dataclass
class CacheNode:
    name: str
    slots: int
    ref_count: int = 0
    last_access: int = 0


class ToyMemoryPool:
    def __init__(self, capacity, cached_nodes):
        self.capacity = capacity
        self.cached_nodes = {node.name: node for node in cached_nodes}
        self.active_slots = 0
        self.evicted = []

    def used(self):
        return self.active_slots + sum(
            node.slots for node in self.cached_nodes.values()
        )

    def free(self):
        return self.capacity - self.used()

    def add_active(self, slots):
        self.active_slots += slots

    def release_active(self, slots):
        self.active_slots = max(0, self.active_slots - slots)

    def pin(self, node_names):
        pinned = []
        for name in node_names:
            node = self.cached_nodes.get(name)
            if node:
                node.ref_count += 1
                pinned.append(name)
        return pinned

    def evict_for(self, slots, protected):
        while self.free() < slots:
            candidates = [
                node for node in self.cached_nodes.values()
                if node.ref_count == 0 and node.name not in protected
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda node: node.last_access)
            del self.cached_nodes[victim.name]
            self.evicted.append(victim.name)
        return self.free() >= slots

    def allocate(self, slots, protected=()):
        if self.free() < slots:
            self.evict_for(slots, set(protected))
        if self.free() < slots:
            return False
        self.active_slots += slots
        return True


class ToyRadixScheduler:
    def __init__(self, memory, max_tokens, max_seqs, max_prefill_chunk):
        self.memory = memory
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_prefill_chunk = max_prefill_chunk
        self.hit_bonus = 0.2
        self.trace = []

    def matched_nodes(self, req):
        if req.hit_len >= 8:
            return ["shared_root"]
        return []

    def priority(self, req, now):
        wait = now - req.arrival_step
        return wait + self.hit_bonus * req.hit_len

    def cleanup(self, running, waiting):
        cleaned = []
        kept_running = []
        for req in running:
            if req.aborted:
                self.memory.release_active(req.active_slots)
                cleaned.append(req.req_id)
                self.trace.append(("cleanup", req.req_id, req.active_slots))
            else:
                kept_running.append(req)
        kept_waiting = []
        for req in waiting:
            if req.aborted:
                cleaned.append(req.req_id)
                self.trace.append(("cleanup_waiting", req.req_id, 0))
            else:
                kept_waiting.append(req)
        return kept_running, kept_waiting, cleaned

    def schedule_step(self, running, waiting, now):
        running, waiting, cleaned = self.cleanup(running, waiting)
        used_tokens = 0
        used_seqs = 0
        decode_rows = []
        prefill_rows = []
        pinned_nodes = []
        grammar_steps = 0
        deferred = {}

        for req in running:
            if used_tokens + 1 > self.max_tokens or used_seqs + 1 > self.max_seqs:
                deferred[req.req_id] = "decode_budget"
                continue
            if not self.memory.allocate(1):
                deferred[req.req_id] = "kv_decode_full"
                continue
            req.generated += 1
            req.phase = "DECODING" if req.generated < req.max_new_tokens else "FINISHED"
            used_tokens += 1
            used_seqs += 1
            if req.grammar:
                grammar_steps += 1
            decode_rows.append(req.req_id)
            self.trace.append(("decode", req.req_id, 1))

        candidates = sorted(
            waiting,
            key=lambda req: (-self.priority(req, now), req.remaining_suffix, req.req_id),
        )
        scheduled_waiting = set()
        for req in candidates:
            if req.remaining_suffix == 0:
                continue
            chunk = min(req.remaining_suffix, self.max_prefill_chunk)
            if used_seqs + 1 > self.max_seqs:
                deferred[req.req_id] = "sequence_budget"
                continue
            if used_tokens + chunk > self.max_tokens:
                deferred[req.req_id] = "token_budget"
                continue
            nodes = self.matched_nodes(req)
            pinned_nodes.extend(self.memory.pin(nodes))
            if not self.memory.allocate(chunk, protected=nodes):
                deferred[req.req_id] = "kv_prefill_full"
                continue
            req.computed_suffix += chunk
            req.phase = "DECODING" if req.remaining_suffix == 0 else "WAITING"
            used_tokens += chunk
            used_seqs += 1
            scheduled_waiting.add(req.req_id)
            prefill_rows.append(
                {
                    "id": req.req_id,
                    "prompt_len": req.prompt_len,
                    "hit_len": req.hit_len,
                    "chunk": chunk,
                    "remaining_suffix": req.remaining_suffix,
                    "priority": round(self.priority(req, now), 2),
                }
            )
            self.trace.append(("prefill", req.req_id, chunk))

        next_waiting = [
            req for req in waiting
            if req.req_id not in scheduled_waiting or req.remaining_suffix > 0
        ]
        summary = {
            "decode_ids": decode_rows,
            "prefill_ids": [row["id"] for row in prefill_rows],
            "scheduled_tokens": used_tokens,
            "scheduled_sequences": used_seqs,
            "prefill_tokens": sum(row["chunk"] for row in prefill_rows),
            "prompt_tokens_if_naive": sum(row["prompt_len"] for row in prefill_rows),
            "grammar_steps": grammar_steps,
            "cleaned_aborts": cleaned,
            "evicted_nodes": list(self.memory.evicted),
            "pinned_nodes": pinned_nodes,
            "kv_free_after": self.memory.free(),
            "deferred": deferred,
        }
        return running, next_waiting, prefill_rows, summary


running = [
    Request(
        "stream_a", 0, 0, 0, max_new_tokens=3,
        phase="DECODING", active_slots=6, grammar=True, stream=True
    ),
    Request(
        "stream_b", 0, 0, 0, max_new_tokens=2,
        phase="DECODING", active_slots=5, stream=True
    ),
    Request(
        "cancel_c", 0, 0, 0, max_new_tokens=4,
        phase="DECODING", active_slots=4, aborted=True
    ),
]
waiting = [
    Request("w_shared", prompt_len=14, hit_len=10, arrival_step=0, max_new_tokens=3),
    Request("w_miss_old", prompt_len=9, hit_len=0, arrival_step=-7, max_new_tokens=2),
    Request(
        "w_json", prompt_len=12, hit_len=8,
        arrival_step=0, max_new_tokens=2, grammar=True
    ),
    Request("w_tiny", prompt_len=3, hit_len=0, arrival_step=0, max_new_tokens=1),
]
cache_nodes = [
    CacheNode("shared_root", slots=10, ref_count=0, last_access=5),
    CacheNode("stale_tail", slots=4, ref_count=0, last_access=1),
    CacheNode("cold_suffix", slots=3, ref_count=0, last_access=2),
]
memory = ToyMemoryPool(capacity=32, cached_nodes=cache_nodes)
for req in running:
    memory.add_active(req.active_slots)

scheduler = ToyRadixScheduler(
    memory=memory,
    max_tokens=10,
    max_seqs=4,
    max_prefill_chunk=4,
)
running, waiting, prefill_rows, summary = scheduler.schedule_step(
    running, waiting, now=1
)
gates = {
    "decode_first": scheduler.trace[:3] == [
        ("cleanup", "cancel_c", 4),
        ("decode", "stream_a", 1),
        ("decode", "stream_b", 1),
    ],
    "token_budget_respected": summary["scheduled_tokens"] <= scheduler.max_tokens,
    "sequence_budget_respected": summary["scheduled_sequences"] <= scheduler.max_seqs,
    "suffix_cost_used": (
        summary["prefill_tokens"] == 8 and summary["prompt_tokens_if_naive"] == 23
    ),
    "kv_eviction_safe": "shared_root" not in summary["evicted_nodes"],
    "aging_prevents_starvation": summary["prefill_ids"][0] == "w_miss_old",
    "chunked_prefill_visible": prefill_rows[0]["remaining_suffix"] == 5,
    "grammar_cost_tracked": summary["grammar_steps"] == 1,
    "abort_cleanup_releases_slots": summary["cleaned_aborts"] == ["cancel_c"],
    "metrics_ready": summary["kv_free_after"] == 1 and bool(summary["deferred"]),
}
gates["sglang_scheduler_gate"] = all(gates.values())

print("sglang_scheduler_prefill_rows=", prefill_rows)
print("sglang_scheduler_summary=", summary)
print("sglang_scheduler_gates=", gates)
```

这段 demo 应该输出：

```text
sglang_scheduler_prefill_rows= [{'id': 'w_miss_old', 'prompt_len': 9, 'hit_len': 0, 'chunk': 4, 'remaining_suffix': 5, 'priority': 8.0}, {'id': 'w_shared', 'prompt_len': 14, 'hit_len': 10, 'chunk': 4, 'remaining_suffix': 0, 'priority': 3.0}]
sglang_scheduler_summary= {'decode_ids': ['stream_a', 'stream_b'], 'prefill_ids': ['w_miss_old', 'w_shared'], 'scheduled_tokens': 10, 'scheduled_sequences': 4, 'prefill_tokens': 8, 'prompt_tokens_if_naive': 23, 'grammar_steps': 1, 'cleaned_aborts': ['cancel_c'], 'evicted_nodes': ['stale_tail', 'cold_suffix'], 'pinned_nodes': ['shared_root'], 'kv_free_after': 1, 'deferred': {'w_json': 'sequence_budget', 'w_tiny': 'sequence_budget'}}
sglang_scheduler_gates= {'decode_first': True, 'token_budget_respected': True, 'sequence_budget_respected': True, 'suffix_cost_used': True, 'kv_eviction_safe': True, 'aging_prevents_starvation': True, 'chunked_prefill_visible': True, 'grammar_cost_tracked': True, 'abort_cleanup_releases_slots': True, 'metrics_ready': True, 'sglang_scheduler_gate': True}
```

读这段代码时要注意：

1. `cancel_c` 先释放 4 个 active KV slots，再调度 `stream_a` 和 `stream_b` decode，说明 cleanup 是调度循环的一部分。
2. `w_shared` 的 prompt length 是 14，但命中 10 个 tokens，本轮只按 4 个 suffix tokens 计入 prefill budget。
3. `w_miss_old` 没有 cache hit，但因为等待时间足够长，优先于 cache hit 请求进入本轮，说明 aging 防止 cache miss 饥饿。
4. `stale_tail` 和 `cold_suffix` 被 eviction，但 `shared_root` 被 pin 后没有被淘汰，说明 KV eviction 必须避开 active / planned prefix。
5. `grammar_steps=1` 说明 structured output 的每步额外开销要进入 scheduler 观测指标，而不是只在 sampler 里悄悄发生。

## 29.28 小练习

1. 画出 SGLang scheduler 的输入、输出和下游模块。
2. 给定 3 个 running 请求和 4 个 waiting 请求，按 token budget 手工做一次调度。
3. 设计一个 cache-aware admission 策略，说明如何避免 cache miss 请求饥饿。
4. 解释为什么 chunked prefill 能降低 streaming 抖动。
5. 说明 structured output 为什么会影响 scheduler 的成本估计。
6. 给出 KV cache 不够时的 4 种处理策略，并说明优缺点。
7. 设计一个 SGLang scheduler dashboard，包含 waiting、running、hit length、scheduled prefill tokens、scheduled decode tokens、eviction 和 p99。

## 29.29 本章总结

SGLang scheduler 的核心不是“把请求凑成 batch”，而是在每个 engine step 做多目标资源分配。

它既继承了 LLM serving 中 continuous batching 和 iteration-level scheduling 的基本思想，也加入了 SGLang 特有的 RadixAttention、cache-aware scheduling、frontend program 分支和 structured output 约束。

理解 scheduler 后，就能把第 27 章的 runtime 总览和第 28 章的 RadixAttention 串起来：RadixAttention 提供可复用 prefix，scheduler 决定何时复用、复用谁、计算多少 suffix、是否淘汰 cache、如何平衡 prefill 与 decode。下一章会继续深入 structured generation 与 constrained decoding，解释 JSON schema、regex、EBNF 和 grammar mask 如何进入每一步采样。
