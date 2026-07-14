# 第 52 章 把调度、KV cache、prefix cache 和 preemption 串起来

前面几章我们按模块升级了 mini engine。

第 48 章把 naive scheduler 升级成 continuous batching。

第 49 章把 request 私有 list KV cache 升级成 paged KV cache。

第 50 章在 paged KV cache 上实现了 prefix cache。

第 51 章补上了 KV 空间不足时的 preemption、recompute 和 swap。

这些机制单独看都不难。

真正容易出问题的是：

```text
它们在同一个 engine step 里怎么配合？
```

本章就把这些机制串成一个完整调度循环。

## 52.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲教学版 vLLM-like engine step 如何把 continuous batching、paged KV cache、prefix cache、chunked prefill 和 preemption 串成一个统一调度循环。

公开资料校准主要参考四类口径：

1. vLLM scheduler / optimization 文档对 decode 优先、chunked prefill、`max_num_batched_tokens`、`max_num_seqs`、KV cache usage 和 preemption 的公开说明。
2. vLLM prefix caching 设计文档对 `KVCacheBlock`、block hash、parent hash、ref count、cached-free block、free queue 和 eviction 的公开说明。
3. vLLM / PagedAttention 论文对 block-based KV cache、logical block 到 physical block 映射、动态 serving request 和高吞吐调度的基础抽象。
4. 本书第 48 到 51 章对 continuous batching、paged KV cache、prefix cache、preemption、recompute 和 swap 的教学版实现边界。

本章不复刻某个框架版本的源码字段、线程模型、CUDA kernel、multi-worker 通信、KV connector、真实 swap 后端、分布式 prefix cache 或完整生产优先级策略。我们只验证一个最小闭环：

```text
waiting/running 队列 -> prefix lookup once -> decode-first -> suffix prefill -> token/KV budget -> eviction/preemption -> commit plan -> BatchBuilder metadata -> OutputProcessor 状态更新 -> invariants
```

第二轮新增 demo 的验收重点是：

```text
prefix cache lookup 是否只对 waiting 请求做一次；
decode 是否先拿到预算，避免 streaming TPOT 抖动；
prefix hit 后是否只调度 suffix prefill；
token budget 和 KV block budget 是否同时生效；
KV 压力下是否先 evict cached-free block，再 preempt active request；
commit plan 是否先分配 block，再让 BatchBuilder 读 metadata；
positions、slot mapping、block table 是否按真实上下文位置构造；
OutputProcessor 是否正确更新 output token 和 num_computed_tokens；
队列、block、computed token、metadata 和 cleanup 不变量是否能被检查。
```

## 52.1 本章目标

读完本章，你应该能讲清：

1. 一个 vLLM-like engine step 里 scheduler 大致做哪些事。
2. waiting、running、preempted、finished 请求如何流转。
3. prefix cache lookup 应该发生在什么时候。
4. KV block 预算应该在调度前还是调度后计算。
5. decode、prefill、chunked prefill 的优先级如何取舍。
6. free blocks 不够时，cache eviction 和 preemption 的顺序如何安排。
7. BatchBuilder 如何把 scheduler 决策变成模型可执行的 batch metadata。
8. ModelRunner、Sampler、OutputProcessor 如何更新 request 状态。
9. 哪些日志和断言能帮助定位调度 bug。
10. 面试里如何描述一个完整 serving engine 的主循环。

这一章不是引入新概念。

它的目标是把前面所有零件装成一台能转起来的机器。

## 52.2 先看完整循环

一个最小 engine loop 可以抽象成：

```python
while engine_has_work():
    new_requests = api_server.pop_new_requests()
    scheduler.add_requests(new_requests)

    schedule = scheduler.schedule()
    if schedule.is_empty():
        wait_for_new_work()
        continue

    batch = batch_builder.build(schedule)
    model_outputs = model_runner.forward(batch)
    sampled_tokens = sampler.sample(model_outputs, batch)
    output_processor.process(schedule, sampled_tokens)
```

这段代码看起来简单。

但复杂度都藏在 `scheduler.schedule()` 和 `output_processor.process()` 里。

其中 scheduler 负责回答：

```text
这一轮哪些请求可以执行？每个请求执行 prefill 还是 decode？需要多少 token budget 和 KV blocks？
```

OutputProcessor 负责回答：

```text
模型执行完之后，请求状态、KV blocks、prefix cache、streaming 输出和 metrics 怎么更新？
```

## 52.3 request 队列

先定义几个队列。

```python
class Scheduler:
    def __init__(self):
        self.waiting = deque()
        self.running = []
        self.swapped = deque()
        self.finished = []
```

这里 `waiting` 包含三类请求：

1. 新进入 engine、还没 prefill 的请求。
2. recompute preemption 后等待恢复的请求。
3. prefix cache lookup 完成但还没被调度的请求。

`running` 是当前持有 GPU KV、可以继续 prefill 或 decode 的请求。

`swapped` 是 KV 在 CPU 上、等待 swap in 的请求。

`finished` 通常不会长期留在 scheduler 里，只是为了本轮统一清理和统计。

教学版如果不实现 swap，可以先没有 `swapped`。

## 52.4 request 的最小状态

request 需要保存逻辑状态和执行状态。

```python
class RequestState:
    def __init__(self, request_id, prompt_token_ids, sampling_params):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = []
        self.sampling_params = sampling_params

        self.status = RequestStatus.WAITING
        self.block_table = []
        self.num_computed_tokens = 0

        self.prefix_hit_tokens = 0
        self.preempted_count = 0
        self.arrival_time = now_ms()
        self.last_scheduled_time = None
```

这里最关键的字段是 `num_computed_tokens`。

它表示：

```text
当前 request 已经有多少上下文 tokens 的 KV 可用。
```

不同机制都会修改它：

1. 新请求初始为 0。
2. prefix cache 命中后变成 hit tokens。
3. prefill chunk 执行后增加 chunk size。
4. decode 生成一个 token 后增加 1。
5. recompute preemption 后重置为 0，或者重新经过 prefix cache lookup 后变成 hit tokens。

只要 `num_computed_tokens` 维护错，后面 positions、slot mapping、attention metadata 都会错。

## 52.5 scheduler 每轮要做的事

一个完整的 `schedule()` 可以分成八步：

```text
1. 清理已经 finished / aborted 的请求。
2. 尝试 swap in 或恢复 preempted 请求。
3. 对新 waiting 请求做 prefix cache lookup。
4. 预留 running decode 的 KV 增长预算。
5. 在 token budget 内选择 prefill chunks。
6. 如果 KV blocks 不够，先 cache eviction，再 preemption。
7. 生成本轮 schedule plan。
8. 更新被选中请求的临时调度状态。
```

不要把这些步骤写成一团。

工程上最容易 debug 的方式是让 `schedule()` 返回一个明确的 plan。

```python
class SchedulePlan:
    def __init__(self):
        self.prefill_items = []
        self.decode_items = []
        self.preempted = []
        self.evicted_blocks = []
```

plan 是 scheduler 和 BatchBuilder 之间的契约。

## 52.6 decode 优先还是 prefill 优先

在线 serving 通常会偏向 decode 优先。

原因是 decode 决定 TPOT。

如果已经在 streaming 的请求长时间拿不到 decode step，用户会看到输出卡顿。

所以常见策略是：

```text
先为 running decode 请求预留预算，再用剩余 budget 安排 waiting prefill。
```

伪代码：

```python
decode_items = select_running_decode_requests(running)
remaining_token_budget = max_num_batched_tokens - len(decode_items)
prefill_items = select_prefill_chunks(waiting, remaining_token_budget)
```

但这不是绝对规则。

如果完全偏向 decode，新请求 TTFT 会变差。

如果过度偏向 prefill，已有 streaming 请求 TPOT 会抖。

调度的本质就是在 TTFT、TPOT、吞吐和公平性之间做权衡。

## 52.7 token budget 和 block budget

scheduler 至少要同时管两个 budget。

第一个是 token budget。

```text
本轮最多处理多少 input tokens。
```

它主要由 `max_num_batched_tokens` 控制。

第二个是 block budget。

```text
本轮最多能分配多少新的 KV blocks。
```

它由 block pool 的 free blocks 决定。

两者不能互相替代。

例如 decode 一轮只处理 1 个 token，但如果这个 token 正好跨过 block 边界，就需要新分配一个 block。

又比如 prefix cache 命中 8K tokens，token budget 可能只消耗 suffix 128 tokens，但 block table 已经引用了大量 cached blocks。

所以调度时要同时检查：

```python
if used_tokens + new_tokens > max_num_batched_tokens:
    reject_this_chunk()

if required_new_blocks > free_blocks:
    handle_memory_pressure()
```

## 52.8 prefix cache lookup 的时机

prefix cache lookup 应该在 waiting 请求第一次参与调度前完成。

流程是：

```text
new request -> tokenize -> waiting queue -> prefix cache lookup -> update block_table / num_computed_tokens -> schedule suffix prefill
```

伪代码：

```python
def prepare_waiting_request(req):
    if req.prefix_lookup_done:
        return

    hit_blocks, hit_tokens = block_manager.find_cached_prefix(req.prompt_token_ids)
    if hit_blocks:
        block_manager.add_active_refs(hit_blocks)
        req.block_table.extend(hit_blocks)
        req.num_computed_tokens = hit_tokens
        req.prefix_hit_tokens = hit_tokens

    req.prefix_lookup_done = True
```

prefix cache lookup 不应该每轮都重复做。

否则会引入额外开销，也可能重复增加 ref count。

如果 request 被 recompute preempt，释放了 active refs 后，可以把 `prefix_lookup_done` 重置，让它恢复时重新查一次 prefix cache。

因为这期间 cache 状态可能已经变化。

## 52.9 prefill chunk 怎么选

对 waiting 请求，scheduler 不一定一次 prefill 完。

它可以选择一个 chunk。

```python
def next_prefill_chunk(req, remaining_token_budget):
    context_tokens = req.prompt_token_ids + req.output_token_ids
    start = req.num_computed_tokens
    remaining = len(context_tokens) - start
    chunk_size = min(remaining, remaining_token_budget, max_prefill_chunk_size)
    return start, chunk_size
```

注意这里用的是 `context_tokens`，不是只有 prompt。

对普通新请求，`output_token_ids` 为空。

对 recompute 恢复请求，`output_token_ids` 可能不为空。

因此这一个函数同时覆盖新请求和 recompute 请求。

## 52.10 running 请求何时 decode

一个 request 可以 decode 的前提是：

```text
num_computed_tokens == len(prompt_token_ids) + len(output_token_ids)
```

也就是完整上下文 KV 已经可用。

如果还没 prefill 完，它仍然处于 prefill 阶段。

伪代码：

```python
def is_ready_to_decode(req):
    return req.num_computed_tokens == len(req.prompt_token_ids) + len(req.output_token_ids)
```

decode 后会采样出一个新 token。

这个 token 在下一轮成为上下文的一部分。

所以 OutputProcessor 要做：

```python
req.output_token_ids.append(next_token)
req.num_computed_tokens += 1
```

这里有一个容易混淆的点。

decode step 的输入是上一 token 或最后一个 prompt token，但执行后生成的是新 token 的 logits。

工程实现里 `num_computed_tokens` 如何增加，取决于 KV 写入的是哪个 token 的 KV。

教学版可以采用简化模型：每轮 decode 会为新加入上下文的 token 补齐 KV，并让 `num_computed_tokens` 增加 1。

真实框架里要更严格地区分 input token、sampled token 和下一轮 decode token。

面试时重点讲状态一致性，不必纠缠某个教学伪代码的 off-by-one。

## 52.11 KV block 分配的统一入口

不要让 BatchBuilder、ModelRunner、OutputProcessor 到处直接分配 block。

建议所有 KV 分配都经过 BlockManager。

```python
class BlockManager:
    def ensure_blocks_for_tokens(self, req, target_num_tokens):
        required_blocks = ceil_div(target_num_tokens, self.block_size)

        while len(req.block_table) < required_blocks:
            block_id = self.allocate_one()
            req.block_table.append(block_id)
```

scheduler 在生成 plan 前先做预算。

真正分配可以发生在 schedule commit 阶段。

这样 BatchBuilder 拿到的 request 已经有足够 block table。

```python
def commit_plan(plan):
    for item in plan.prefill_items:
        target = item.start + item.num_tokens
        block_manager.ensure_blocks_for_tokens(item.request, target)

    for item in plan.decode_items:
        target = len(item.request.prompt_token_ids) + len(item.request.output_token_ids) + 1
        block_manager.ensure_blocks_for_tokens(item.request, target)
```

不要等 kernel 执行到一半才发现 block 不够。

## 52.12 memory pressure 处理顺序

当 block budget 不够时，推荐顺序是：

1. 缩小本轮 prefill chunk。
2. 暂缓接纳部分 waiting 请求。
3. 淘汰无 active refs 的 prefix cache blocks。
4. preempt 低优先级 running 请求。
5. 触发入口 backpressure 或拒绝新请求。

伪代码：

```python
def handle_memory_pressure(required_blocks):
    shrink_prefill_plan()

    if enough_blocks(required_blocks):
        return

    block_manager.evict_cached_blocks_until(required_blocks)

    if enough_blocks(required_blocks):
        return

    preempt_until_enough(required_blocks)

    if not enough_blocks(required_blocks):
        enable_backpressure()
```

这不是唯一顺序。

但它体现了一个原则：

```text
先减少新增压力，再释放可牺牲缓存，最后才动 active request。
```

## 52.13 schedule plan 的数据结构

plan item 要明确告诉 BatchBuilder 做什么。

```python
class PrefillItem:
    def __init__(self, request, start, num_tokens):
        self.request = request
        self.start = start
        self.num_tokens = num_tokens

class DecodeItem:
    def __init__(self, request):
        self.request = request

class SchedulePlan:
    def __init__(self):
        self.prefill_items = []
        self.decode_items = []
```

`PrefillItem.start` 很重要。

它可能不是 0。

常见原因：

1. prefix cache 命中了前缀。
2. chunked prefill 已经完成前几个 chunk。
3. recompute 恢复时部分上下文再次命中 prefix cache。

BatchBuilder 必须从 `start` 开始构造 input ids、positions 和 slot mapping。

## 52.14 BatchBuilder 做什么

BatchBuilder 把 plan 转成模型执行需要的 batch。

它通常构造：

1. input token ids。
2. positions。
3. request 到 batch row 的映射。
4. block tables。
5. slot mapping。
6. attention metadata。

对 prefill item：

```python
tokens = context_tokens[item.start : item.start + item.num_tokens]
positions = range(item.start, item.start + item.num_tokens)
```

对 decode item：

```python
tokens = [last_context_token(req)]
positions = [len(req.prompt_token_ids) + len(req.output_token_ids) - 1]
```

不同框架对 decode token 的定义略有不同。

但无论如何，positions 不能因为 batch 拼接而从 0 重置。

这是 BatchBuilder 最容易犯的错误之一。

## 52.15 OutputProcessor 做什么

模型执行后，OutputProcessor 要更新系统状态。

对 prefill item：

```python
req.num_computed_tokens += item.num_tokens

if req.num_computed_tokens == len(req.prompt_token_ids) + len(req.output_token_ids):
    req.status = RequestStatus.RUNNING
```

如果 prefill 完成的是 full block，还可以注册 prefix cache。

```python
block_manager.register_new_full_blocks(req)
```

对 decode item：

```python
token = sampled_tokens[req.request_id]
req.output_token_ids.append(token)
req.num_computed_tokens += 1
stream(token)

if should_stop(req):
    finish_request(req)
```

finish 时要释放 active KV refs。

```python
def finish_request(req):
    block_manager.free_request_blocks(req)
    req.status = RequestStatus.FINISHED
```

如果某些 blocks 已经注册到 prefix cache，它们不会立即回到 free list。

## 52.16 一个完整 schedule 伪代码

把前面拼起来，可以得到一个教学版 scheduler。

```python
def schedule(self):
    self.cleanup_finished()
    self.restore_preempted_requests()

    for req in self.waiting:
        self.prepare_prefix_cache(req)

    plan = SchedulePlan()
    token_budget = self.max_num_batched_tokens

    for req in self.running:
        if self.is_ready_to_decode(req):
            plan.decode_items.append(DecodeItem(req))
            token_budget -= 1

            if token_budget == 0:
                break

    for req in list(self.waiting):
        if token_budget <= 0:
            break

        start, n = self.next_prefill_chunk(req, token_budget)
        if n == 0:
            continue

        plan.prefill_items.append(PrefillItem(req, start, n))
        token_budget -= n

    required_blocks = self.estimate_new_blocks(plan)
    self.handle_memory_pressure(required_blocks, plan)
    self.commit_plan(plan)

    return plan
```

真实系统会复杂很多。

但主干就是这几步。

## 52.17 commit plan 为什么重要

scheduler 不能只“想好”要执行哪些请求。

它还要把状态推进到一个可执行的中间状态。

commit plan 至少要做：

1. 为 plan 中的请求分配需要的 KV blocks。
2. 把 waiting 请求移动到 running。
3. 标记本轮被调度的 token 范围。
4. 记录 last scheduled time。
5. 更新统计指标。

伪代码：

```python
def commit_plan(plan):
    for item in plan.prefill_items:
        req = item.request
        target = item.start + item.num_tokens
        block_manager.ensure_blocks_for_tokens(req, target)
        move_to_running_if_needed(req)
        req.last_scheduled_time = now_ms()

    for item in plan.decode_items:
        req = item.request
        target = len(req.prompt_token_ids) + len(req.output_token_ids) + 1
        block_manager.ensure_blocks_for_tokens(req, target)
        req.last_scheduled_time = now_ms()
```

commit 成功后，BatchBuilder 才能安全读取 block table。

## 52.18 失败要能回滚

如果 commit plan 过程中分配 block 失败怎么办？

最差的实现是分配了一半，然后直接抛异常。

这会留下脏状态。

更好的方式是让分配具备事务感：

```python
allocated = []
try:
    for req in requests:
        new_blocks = block_manager.allocate_for(req)
        allocated.extend((req, new_blocks))
except AllocationError:
    for req, blocks in allocated:
        block_manager.free_blocks_from_request(req, blocks)
    raise
```

教学版可以简单一点：

```text
先 estimate，确认足够后再 allocate。
```

但要知道 estimate 和真实 allocate 之间可能有差异。

比如 cached blocks 的 ref count、并发修改、异步 abort 都可能改变实际 free blocks。

单线程教学引擎里问题不大，多线程真实引擎里必须小心。

## 52.19 状态不变量

复杂 scheduler 必须有 invariants。

建议每轮结束后检查：

1. 一个 request 只能出现在一个队列里。
2. running request 必须有合法 block table。
3. `num_computed_tokens <= len(prompt_token_ids) + len(output_token_ids)`。
4. block ref count 不能为负。
5. free list 里不能包含 active request 正在引用的 block。
6. cached block 如果在 cache map 中，不能同时作为普通 free block。
7. prefix cache hit tokens 必须是 block size 的整数倍，除非明确支持 partial block。
8. preempted request 不能持有 GPU-only active refs。
9. swapped request 不能同时持有 GPU block table 和 CPU block table，除非处于迁移中间态。
10. finished request 不能留在 running 队列。

这些断言比很多日志都有用。

它们可以在 bug 刚出现时立刻失败，而不是等生成结果错乱才发现。

## 52.20 可观测性：每轮调度日志

建议每轮 scheduler 输出一条 debug 级别结构化日志。

字段包括：

| 字段 | 含义 |
|---|---|
| step_id | engine step 编号 |
| num_waiting | waiting 请求数 |
| num_running | running 请求数 |
| num_swapped | swapped 请求数 |
| prefill_reqs | 本轮 prefill 请求数 |
| prefill_tokens | 本轮 prefill tokens |
| decode_reqs | 本轮 decode 请求数 |
| free_blocks_before | 调度前 free blocks |
| free_blocks_after | commit 后 free blocks |
| cached_blocks | prefix cache blocks |
| evicted_blocks | 本轮 eviction blocks |
| preempted_reqs | 本轮 preempt 请求数 |

示例：

```text
schedule step=1024 waiting=8 running=64 prefill_reqs=2 prefill_tokens=512 \
decode_reqs=62 free_blocks_before=120 free_blocks_after=88 cached_blocks=300 \
evicted_blocks=0 preempted_reqs=0
```

当 TTFT 或 TPOT 抖动时，这条日志能快速告诉你：

1. 是 waiting 太多。
2. 还是 decode 太少。
3. 还是 free blocks 快没了。
4. 还是 preemption/eviction 在频繁发生。

## 52.21 指标分层

不要只暴露一个总吞吐。

至少分四层指标。

第一层，请求层：

1. QPS。
2. TTFT p50/p99。
3. TPOT p50/p99。
4. E2E latency p50/p99。
5. active requests。
6. waiting queue length。

第二层，调度层：

1. scheduled_prefill_tokens。
2. scheduled_decode_tokens。
3. max_num_batched_tokens utilization。
4. running requests。
5. preemptions。
6. chunked prefill count。

第三层，KV 层：

1. free blocks。
2. used blocks。
3. cached blocks。
4. allocation failures。
5. block reuse rate。
6. eviction count。

第四层，模型执行层：

1. forward latency。
2. GPU utilization。
3. input tokens/s。
4. output tokens/s。
5. kernel time。
6. sampling time。

分层之后，问题定位会清楚很多。

## 52.22 压测场景一：稳定短请求

workload：

```text
prompt 128 tokens
output 128 tokens
固定并发
无共享 prefix
```

预期：

1. waiting queue 稳定。
2. preemption 为 0。
3. prefix cache hit ratio 接近 0。
4. free blocks 在稳定区间波动。
5. decode batch size 接近 running 请求数。

这个场景是 baseline。

如果 baseline 都不稳定，不要急着测复杂场景。

## 52.23 压测场景二：共享 system prompt

workload：

```text
system prompt 2K tokens 固定
user prompt 64 tokens 随机
output 128 tokens
```

预期：

1. prefix cache hit tokens 上升。
2. TTFT 下降。
3. scheduled_prefill_tokens 中真实计算部分下降。
4. cached blocks 上升后趋于稳定。
5. preemption 不应该明显增加。

如果 TTFT 没下降，检查 prefix cache lookup、ref count、suffix prefill start position。

如果 preemption 增加，检查 cached block eviction 水位线。

## 52.24 压测场景三：长上下文混合短请求

workload：

```text
80% 请求：prompt 128 tokens，output 128 tokens
20% 请求：prompt 16K tokens，output 512 tokens
```

预期：

1. chunked prefill 生效。
2. 短请求 TTFT 不被长请求完全拖垮。
3. prefill tokens 和 decode tokens 在每轮都有合理比例。
4. free blocks 下降但不频繁归零。
5. preemption 只在压力峰值偶发。

这个场景最能验证完整 scheduler 的工程质量。

## 52.25 压测场景四：故意打爆 KV

workload：

```text
逐步增加并发和 max output tokens
直到 KV block pool 接近耗尽
```

预期不是“永远不 preempt”。

预期是：

1. allocation failures 不导致 engine crash。
2. 先出现 chunk 缩小或 waiting 增长。
3. 然后出现 cache eviction。
4. 最后才出现 preemption。
5. preempted request 能恢复并完成，或被明确 abort。

这个场景验证鲁棒性。

不要只测性能好看的区间。

## 52.26 常见 bug

bug 一：prefix cache lookup 每轮重复执行。

```text
结果：ref count 重复增加，cached block 永远无法释放。
```

bug 二：chunked prefill 的 positions 从 0 开始。

```text
结果：RoPE position 错误，长 prompt 输出异常。
```

bug 三：recompute 请求只 prefill prompt。

```text
结果：已生成 output tokens 没有进入 KV，上下文断裂。
```

bug 四：commit plan 后 waiting/running 队列重复持有同一请求。

```text
结果：同一个 request 被重复调度。
```

bug 五：估算 block 足够，但实际分配失败后没有回滚。

```text
结果：block table 半更新，free list 和 ref count 不一致。
```

bug 六：finish request 时把 cached block 放回 free list。

```text
结果：prefix cache 悬挂引用。
```

bug 七：过度 decode 优先。

```text
结果：TPOT 稳定但新请求 TTFT 很差。
```

bug 八：过度 prefill 优先。

```text
结果：TTFT 看似不错，但 streaming 输出卡顿。
```

## 52.27 面试高频问题

问题一：一个 serving engine 的 scheduler 每轮大概做什么？

回答要点：每轮在 iteration boundary 上重新选择 batch。它会处理 waiting/running 请求，做 prefix cache lookup，优先安排 running decode，再用剩余 token budget 安排 prefill chunk，同时检查 KV block budget。如果 free blocks 不够，先缩小 prefill 或推迟请求，再做 cache eviction，最后 preempt running 请求。最后生成 schedule plan 给 BatchBuilder。

问题二：为什么 scheduler 要同时看 token budget 和 KV block budget？

回答要点：token budget 控制本轮计算量，KV block budget 控制显存资源。decode 可能只加一个 token 但跨 block 边界需要新 block；prefix cache 命中会减少计算 tokens，但仍引用 KV blocks。两者约束不同，必须同时管理。

问题三：prefix cache lookup 放在哪里？

回答要点：通常在 waiting 请求第一次参与调度前做。命中后更新 request 的 block table、active refs、`num_computed_tokens` 和 hit tokens，然后 scheduler 只调度 suffix prefill。不能每轮重复 lookup，否则 ref count 和开销都会出问题。

问题四：BatchBuilder 最容易出什么错？

回答要点：最容易把 positions 从 0 重置，或者忽略 prefill chunk 的 start offset。prefix cache、chunked prefill、recompute 都可能导致 prefill 从非零位置开始，因此 input ids、positions、slot mapping 和 block table 必须按真实上下文位置构造。

问题五：如何判断调度策略有问题？

回答要点：要看 TTFT、TPOT、E2E latency、waiting queue、running queue、scheduled prefill/decode tokens、free blocks、cached blocks、evictions、preemptions 和 allocation failures。如果 TTFT 高，可能新请求进不来或 prefill budget 太小；如果 TPOT 抖，可能 decode 被 prefill 挤压；如果 preemption 高，说明 KV budget 或并发参数有问题。

## 52.28 标准回答模板

如果面试官问“你怎么设计一个 vLLM-like scheduler 主循环”，可以这样回答：

```text
我会把 engine 看成 iteration-level 调度系统。每个 engine step 在 GPU forward 结束后重新调度一次，scheduler 维护 waiting、running、swapped 或 preempted 请求队列。新请求进入 waiting 后先 tokenize，真正参与调度前做 prefix cache lookup；如果命中，就把 cached blocks 加到 block table，增加 active refs，并把 num_computed_tokens 设置为 hit tokens。

每轮调度时，我会优先考虑 running 请求的 decode，保证 streaming TPOT 不抖；然后用剩余 max_num_batched_tokens 选择 waiting 请求的 prefill chunks。这里要同时管理 token budget 和 KV block budget。token budget 控制本轮计算量，block budget 控制 KV cache 显存。调度前会估算本轮新增 blocks，commit plan 时统一通过 BlockManager 分配，避免执行中才发现 OOM。

如果 KV blocks 不够，我不会直接 OOM。处理顺序一般是缩小本轮 prefill chunk、推迟部分 waiting 请求、淘汰无 active refs 的 prefix cache blocks；还不够时再 preempt 低优先级或 recompute 成本低的 running 请求。preemption 后保留 prompt tokens 和已生成 output tokens，释放 GPU KV，恢复时重新 prefill prompt + output tokens。

BatchBuilder 根据 schedule plan 构造 input ids、positions、slot mapping、block tables 和 attention metadata。这里最关键的是不要把 positions 重置为 0，因为 prefix cache、chunked prefill 和 recompute 都可能让 prefill 从非零位置开始。ModelRunner 执行后，OutputProcessor 更新 num_computed_tokens、追加 sampled tokens、注册 prefix cache blocks、释放 finished 请求的 active refs，并记录 TTFT、TPOT、free blocks、eviction 和 preemption 指标。

最后我会用 invariants 和分层指标保证系统可 debug，比如一个 request 只能在一个队列里，ref count 不能为负，free list 不能包含 active block，cached block 不能被误释放。压测会覆盖短请求 baseline、共享 prefix、长短混合和故意打爆 KV 的场景。
```

## 52.29 统一调度循环公式、状态不变量和可运行 demo

把本章机制合起来后，可以先把每轮调度的 token 用量写成：

```math
U_t=|D_t|+\sum_{i\in P_t}c_{i,t}
```

其中 `D_t` 是本轮 decode 请求集合，`P_t` 是本轮 prefill 请求集合，`c_{i,t}` 是请求 `i` 本轮 prefill chunk tokens。

token budget 约束是：

```math
U_t\le B_{\mathrm{tok}}
```

新增 KV block 需求是：

```math
K_t^{\mathrm{need}}=\sum_{i\in D_t\cup P_t}\max(0,\left\lceil\frac{T_{i,t}^{\mathrm{target}}}{S}\right\rceil-|P_i|)
```

其中 `S` 是 block size，`T_{i,t}^{\mathrm{target}}` 是本轮执行后请求 `i` 至少需要覆盖到的上下文长度，`|P_i|` 是当前 block table 长度。

KV 压力门可以写成：

```math
G_{\mathrm{pressure}}=\mathbf{1}[K_t^{\mathrm{need}}>F_t]
```

其中 `F_t` 是调度前可直接分配的 free blocks。

prefix 命中后的剩余 prefill token 数是：

```math
R_i^{\mathrm{prefill}}=T_i-T_i^{\mathrm{hit}}
```

每轮不变量门禁可以拆成：

```math
G_{\mathrm{invariant}}=G_{\mathrm{queue}}G_{\mathrm{block}}G_{\mathrm{computed}}G_{\mathrm{metadata}}G_{\mathrm{cleanup}}
```

最终统一调度循环门禁：

```math
G_{\mathrm{loop}}=G_{\mathrm{prefix}}G_{\mathrm{decode}}G_{\mathrm{prefill}}G_{\mathrm{budget}}G_{\mathrm{pressure}}G_{\mathrm{commit}}G_{\mathrm{metadata}}G_{\mathrm{output}}G_{\mathrm{invariant}}
```

下面这个 0 依赖 demo 模拟一个完整 engine step：

1. `cache_hit` 只做一次 prefix lookup，命中 4 个 tokens。
2. `stream` 作为 running streaming 请求优先 decode。
3. `cache_hit` 只 prefill suffix，`cache_miss` 从 0 开始 prefill。
4. 本轮需要 3 个新 KV blocks，但 free list 只有 1 个。
5. scheduler 先 evict cached-free block，再 preempt 低优先级 running 请求。
6. commit plan 统一分配 blocks，BatchBuilder 再构造 positions、slot mapping 和 block tables。
7. OutputProcessor 追加 decode token，推进 prefill computed tokens，并检查状态不变量。

```python
from dataclasses import dataclass, field


def ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class RequestState:
    request_id: str
    prompt_tokens: list
    output_tokens: list = field(default_factory=list)
    priority: int = 0
    status: str = "WAITING"
    block_table: list = field(default_factory=list)
    num_computed_tokens: int = 0
    prefix_lookup_done: bool = False
    prefix_hit_tokens: int = 0
    want_decode: bool = False
    preempted_count: int = 0

    def context_tokens(self):
        return self.prompt_tokens + self.output_tokens


@dataclass
class PrefillItem:
    request: RequestState
    start: int
    num_tokens: int


@dataclass
class DecodeItem:
    request: RequestState


@dataclass
class SchedulePlan:
    prefill_items: list = field(default_factory=list)
    decode_items: list = field(default_factory=list)
    evicted_blocks: list = field(default_factory=list)
    preempted: list = field(default_factory=list)
    commit_allocations: dict = field(default_factory=dict)


class ToyUnifiedBlockManager:
    def __init__(self, block_size):
        self.block_size = block_size
        self.free_blocks = [7]
        self.cached_free_blocks = [6]
        self.prefix_cache = {"shared": [5]}
        self.active_ref = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0}

    def required_blocks(self, token_count):
        return ceil_div(token_count, self.block_size)

    def physical_slot(self, req, position):
        block_index = position // self.block_size
        offset = position % self.block_size
        return req.block_table[block_index] * self.block_size + offset

    def lookup_prefix_once(self, req):
        if req.prefix_lookup_done:
            return req.prefix_hit_tokens
        if req.request_id == "cache_hit":
            hit_blocks = list(self.prefix_cache["shared"])
            req.block_table.extend(hit_blocks)
            req.num_computed_tokens = self.block_size
            req.prefix_hit_tokens = self.block_size
            for block_id in hit_blocks:
                self.active_ref[block_id] = self.active_ref.get(block_id, 0) + 1
        req.prefix_lookup_done = True
        return req.prefix_hit_tokens

    def estimate_new_blocks(self, plan):
        total = 0
        for item in plan.decode_items:
            req = item.request
            target = len(req.context_tokens()) + 1
            total += max(0, self.required_blocks(target) - len(req.block_table))
        for item in plan.prefill_items:
            req = item.request
            target = item.start + item.num_tokens
            total += max(0, self.required_blocks(target) - len(req.block_table))
        return total

    def evict_cached_free_until_visible(self):
        if not self.cached_free_blocks:
            return []
        block_id = self.cached_free_blocks.pop(0)
        self.free_blocks.append(block_id)
        return [block_id]

    def free_request_blocks(self, req):
        released = list(req.block_table)
        for block_id in released:
            self.active_ref[block_id] = max(0, self.active_ref.get(block_id, 1) - 1)
        self.free_blocks.extend(released)
        req.block_table = []
        req.num_computed_tokens = 0
        return released

    def allocate_until(self, req, target_tokens):
        required = self.required_blocks(target_tokens)
        allocated = []
        while len(req.block_table) < required:
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)
            self.active_ref[block_id] = self.active_ref.get(block_id, 0) + 1
            allocated.append(block_id)
        return allocated


class ToyUnifiedScheduler:
    def __init__(self, block_mgr, waiting, running, max_tokens):
        self.block_mgr = block_mgr
        self.waiting = list(waiting)
        self.running = list(running)
        self.preempted = []
        self.max_tokens = max_tokens
        self.prefix_hits = {}
        self.free_before = 0
        self.required_new_blocks = 0

    def ready_to_decode(self, req):
        return req.want_decode and req.num_computed_tokens == len(req.context_tokens())

    def next_prefill_chunk(self, req, token_budget):
        start = req.num_computed_tokens
        remaining = len(req.context_tokens()) - start
        return start, min(remaining, token_budget)

    def preempt_low_priority(self):
        candidates = [req for req in self.running if not req.want_decode]
        victim = min(candidates, key=lambda req: (req.priority, req.preempted_count))
        context = victim.context_tokens()
        released = self.block_mgr.free_request_blocks(victim)
        victim.status = "PREEMPTED"
        victim.preempted_count += 1
        self.running.remove(victim)
        self.preempted.append(victim)
        return victim.request_id, released, context

    def schedule(self):
        for req in self.waiting:
            self.prefix_hits[req.request_id] = self.block_mgr.lookup_prefix_once(req)

        plan = SchedulePlan()
        used_tokens = 0
        for req in self.running:
            if self.ready_to_decode(req):
                plan.decode_items.append(DecodeItem(req))
                used_tokens += 1

        for req in list(self.waiting):
            if used_tokens >= self.max_tokens:
                break
            start, chunk = self.next_prefill_chunk(req, self.max_tokens - used_tokens)
            if chunk:
                plan.prefill_items.append(PrefillItem(req, start, chunk))
                used_tokens += chunk

        self.free_before = len(self.block_mgr.free_blocks)
        self.required_new_blocks = self.block_mgr.estimate_new_blocks(plan)
        if self.required_new_blocks > len(self.block_mgr.free_blocks):
            plan.evicted_blocks = self.block_mgr.evict_cached_free_until_visible()
        if self.required_new_blocks > len(self.block_mgr.free_blocks):
            plan.preempted.append(self.preempt_low_priority())

        for item in plan.decode_items:
            req = item.request
            target = len(req.context_tokens()) + 1
            plan.commit_allocations[req.request_id] = self.block_mgr.allocate_until(req, target)
        for item in plan.prefill_items:
            req = item.request
            target = item.start + item.num_tokens
            plan.commit_allocations[req.request_id] = self.block_mgr.allocate_until(req, target)
            req.status = "RUNNING"
            if req in self.waiting:
                self.waiting.remove(req)
                self.running.append(req)
        return plan


class ToyBatchBuilder:
    def __init__(self, block_mgr):
        self.block_mgr = block_mgr

    def build(self, plan):
        positions = []
        slot_mapping = []
        block_tables = []
        for item in plan.decode_items:
            req = item.request
            position = len(req.context_tokens())
            positions.append(position)
            slot_mapping.append(self.block_mgr.physical_slot(req, position))
            block_tables.append((req.request_id, list(req.block_table)))
        for item in plan.prefill_items:
            req = item.request
            for position in range(item.start, item.start + item.num_tokens):
                positions.append(position)
                slot_mapping.append(self.block_mgr.physical_slot(req, position))
            block_tables.append((req.request_id, list(req.block_table)))
        return {
            "positions": positions,
            "slot_mapping": slot_mapping,
            "block_tables": block_tables,
        }


class ToyOutputProcessor:
    def process(self, plan, sampled_tokens):
        for item in plan.decode_items:
            req = item.request
            req.output_tokens.append(sampled_tokens[req.request_id])
            req.num_computed_tokens += 1
        for item in plan.prefill_items:
            req = item.request
            req.num_computed_tokens = item.start + item.num_tokens


def check_invariants(scheduler, metadata, block_mgr):
    queues = {
        "waiting": scheduler.waiting,
        "running": scheduler.running,
        "preempted": scheduler.preempted,
    }
    memberships = []
    for queue_name, queue in queues.items():
        memberships.extend((req.request_id, queue_name) for req in queue)
    request_ids = [item[0] for item in memberships]
    active_blocks = []
    for req in scheduler.running:
        active_blocks.extend(req.block_table)
    all_requests = scheduler.waiting + scheduler.running + scheduler.preempted
    return {
        "queue_ownership": len(request_ids) == len(set(request_ids)),
        "no_duplicate_active_blocks": len(active_blocks) == len(set(active_blocks)),
        "computed_not_beyond_context": all(
            req.num_computed_tokens <= len(req.context_tokens()) for req in all_requests
        ),
        "free_blocks_not_active": set(block_mgr.free_blocks).isdisjoint(active_blocks),
        "prefix_hit_block_aligned": all(
            req.prefix_hit_tokens % block_mgr.block_size == 0
            for req in all_requests
            if req.prefix_hit_tokens
        ),
        "metadata_rows_consistent": (
            len(metadata["positions"]) == len(metadata["slot_mapping"])
            and len(metadata["block_tables"]) == 3
        ),
        "preempted_cleanup_done": all(not req.block_table for req in scheduler.preempted),
    }


stream = RequestState(
    "stream",
    prompt_tokens=[1, 2, 3, 4],
    output_tokens=[11, 12],
    status="RUNNING",
    block_table=[0, 1],
    num_computed_tokens=6,
    priority=10,
    want_decode=True,
)
low = RequestState(
    "low",
    prompt_tokens=[5, 6, 7, 8, 9, 10, 11, 12],
    output_tokens=[21, 22],
    status="RUNNING",
    block_table=[2, 3, 4],
    num_computed_tokens=10,
    priority=0,
)
cache_hit = RequestState("cache_hit", prompt_tokens=[101, 102, 103, 104, 105, 106])
cache_miss = RequestState("cache_miss", prompt_tokens=[201, 202, 203, 204, 205, 206, 207, 208])

block_mgr = ToyUnifiedBlockManager(block_size=4)
scheduler = ToyUnifiedScheduler(
    block_mgr=block_mgr,
    waiting=[cache_hit, cache_miss],
    running=[stream, low],
    max_tokens=12,
)
plan = scheduler.schedule()
metadata = ToyBatchBuilder(block_mgr).build(plan)
ToyOutputProcessor().process(plan, {"stream": 99})
invariants = check_invariants(scheduler, metadata, block_mgr)

summary = {
    "prefix_hits": scheduler.prefix_hits,
    "final_decode_items": [item.request.request_id for item in plan.decode_items],
    "final_prefill_items": [
        (item.request.request_id, item.start, item.num_tokens) for item in plan.prefill_items
    ],
    "free_before": scheduler.free_before,
    "required_new_blocks": scheduler.required_new_blocks,
    "evicted_blocks": plan.evicted_blocks,
    "preempted": plan.preempted,
    "commit_allocations": plan.commit_allocations,
    "metadata_positions": metadata["positions"],
    "metadata_slot_mapping": metadata["slot_mapping"],
    "metadata_block_tables": metadata["block_tables"],
    "stream_output_tokens": stream.output_tokens,
    "cache_hit_computed": cache_hit.num_computed_tokens,
    "cache_miss_computed": cache_miss.num_computed_tokens,
    "free_after": len(block_mgr.free_blocks),
    "invariants": invariants,
}
gates = {
    "prefix_lookup_once_ready": scheduler.prefix_hits == {"cache_hit": 4, "cache_miss": 0},
    "decode_first_ready": summary["final_decode_items"] == ["stream"],
    "suffix_prefill_ready": summary["final_prefill_items"][0] == ("cache_hit", 4, 2),
    "kv_budget_ready": scheduler.required_new_blocks == 3 and scheduler.free_before == 1,
    "memory_pressure_order_ready": plan.evicted_blocks == [6] and plan.preempted[0][0] == "low",
    "commit_plan_ready": plan.commit_allocations == {
        "stream": [],
        "cache_hit": [7],
        "cache_miss": [6, 2],
    },
    "batch_metadata_ready": metadata["slot_mapping"] == [6, 28, 29, 24, 25, 26, 27, 8, 9, 10, 11],
    "output_processor_ready": stream.output_tokens == [11, 12, 99]
    and cache_hit.num_computed_tokens == 6
    and cache_miss.num_computed_tokens == 8,
    "invariants_ready": all(invariants.values()),
}
gates["unified_scheduler_loop_gate"] = all(gates.values())

print("unified_scheduler_loop_summary=", summary)
print("unified_scheduler_loop_gates=", gates)
```

一次运行的核心输出类似：

```text
unified_scheduler_loop_summary= {'prefix_hits': {'cache_hit': 4, 'cache_miss': 0}, 'final_decode_items': ['stream'], 'final_prefill_items': [('cache_hit', 4, 2), ('cache_miss', 0, 8)], 'free_before': 1, 'required_new_blocks': 3, 'evicted_blocks': [6], 'preempted': [('low', [2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12, 21, 22])], 'commit_allocations': {'stream': [], 'cache_hit': [7], 'cache_miss': [6, 2]}, 'metadata_positions': [6, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7], 'metadata_slot_mapping': [6, 28, 29, 24, 25, 26, 27, 8, 9, 10, 11], 'metadata_block_tables': [('stream', [0, 1]), ('cache_hit', [5, 7]), ('cache_miss', [6, 2])], 'stream_output_tokens': [11, 12, 99], 'cache_hit_computed': 6, 'cache_miss_computed': 8, 'free_after': 2, 'invariants': {'queue_ownership': True, 'no_duplicate_active_blocks': True, 'computed_not_beyond_context': True, 'free_blocks_not_active': True, 'prefix_hit_block_aligned': True, 'metadata_rows_consistent': True, 'preempted_cleanup_done': True}}
unified_scheduler_loop_gates= {'prefix_lookup_once_ready': True, 'decode_first_ready': True, 'suffix_prefill_ready': True, 'kv_budget_ready': True, 'memory_pressure_order_ready': True, 'commit_plan_ready': True, 'batch_metadata_ready': True, 'output_processor_ready': True, 'invariants_ready': True, 'unified_scheduler_loop_gate': True}
```

这个 demo 证明了几个关键点：

1. prefix cache lookup 是 waiting 请求进入调度前的状态推进，不是每轮重复查。
2. decode-first、suffix prefill、token budget 和 KV block budget 必须在同一个 plan 里收敛。
3. KV 压力处理顺序影响线上延迟：先动 cached-free block，再动 active request。
4. BatchBuilder 依赖 commit 后的 block table，不能自己临时分配 block。
5. OutputProcessor 负责把模型输出写回 request 状态，并让下一轮 scheduler 看到一致状态。
6. 统一 loop 的验收不是只看输出 token，而是同时看队列、blocks、positions、slot mapping、computed tokens、cleanup 和 metrics。

## 52.30 小练习

1. 定义 `SchedulePlan`、`PrefillItem` 和 `DecodeItem`。
2. 给 scheduler 增加 waiting、running、swapped 队列。
3. 实现 `prepare_prefix_cache(req)`，确保只执行一次 lookup。
4. 实现 `next_prefill_chunk(req, token_budget)`。
5. 实现 `is_ready_to_decode(req)`。
6. 实现 decode 优先的 `schedule()`。
7. 给 schedule 增加 token budget 检查。
8. 给 schedule 增加 block budget 估算。
9. 实现 `commit_plan(plan)`，统一分配 KV blocks。
10. 实现 memory pressure 处理顺序：缩小 chunk、evict cache、preempt request。
11. 实现 BatchBuilder 的 prefill positions，验证 start 不为 0。
12. 实现 OutputProcessor 更新 `num_computed_tokens`。
13. finish request 时释放 active refs，并保留 cached blocks。
14. 增加每轮 schedule debug 日志。
15. 增加状态不变量检查。
16. 构造短请求 baseline 压测。
17. 构造共享 system prompt 压测。
18. 构造长短混合压测。
19. 构造 KV block pool 打爆压测。
20. 写一段面试回答：scheduler 如何同时管理 token budget 和 KV budget？

## 52.31 本章总结

到这一章，mini engine 的主干已经成型。

continuous batching 负责在 iteration boundary 上动态组织请求。

paged KV cache 让 KV 显存变成可分配、可释放、可统计的 block pool。

prefix cache 让新请求可以复用已有 full blocks，只计算 suffix。

preemption 让系统在 KV 压力下不会轻易 OOM。

这些机制必须通过 scheduler 主循环串起来。

真正的难点不是某个单点算法，而是状态一致性：队列状态、`num_computed_tokens`、block table、ref count、positions、slot mapping、cache entry 和 metrics 必须一起正确。

工程上要用明确的 schedule plan、统一的 BlockManager 入口、commit 阶段、状态不变量、结构化日志和分层指标来控制复杂度。

下一章可以继续讨论如何把这个 mini engine 变成一个可压测、可调参、可回归的实验框架，让每次优化都有数据支撑。
