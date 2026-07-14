# 第 48 章 从 naive scheduler 升级到 continuous batching

上一章我们从 tiny-llm、nano-vLLM、mini-sglang 中抽象出一个最小推理框架骨架：`Request`、`Scheduler`、`KVCacheManager`、`BatchBuilder`、`ModelRunner`、`Sampler`、`OutputProcessor` 和 `Metrics`。

从这一章开始，我们不再只是画模块图，而是对这个 mini engine 做第一轮升级：

```text
把 naive scheduler 升级成 continuous batching scheduler。
```

这是推理框架最重要的升级之一。

原因很简单：

```text
LLM serving 的吞吐和延迟，很大程度取决于 scheduler 如何把不同阶段、不同长度、不同到达时间的请求组织进每一轮 model forward。
```

如果 scheduler 设计不好，即使模型 kernel 很快，系统也会出现 GPU 空转、decode 抖动、长 prompt 阻塞、batch 利用率低、TTFT 和 TPOT p99 很差等问题。

## 48.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲把教学版 `naive scheduler` 升级成单机、单模型、同步 engine loop 内的 continuous batching scheduler。

公开资料校准主要参考三类口径：

1. Orca 论文对 iteration-level scheduling 和 selective batching 的系统动机：生成式模型的 batch 不应该被 request lifetime 锁死，而应该在 iteration 边界动态调度。
2. vLLM 文档对 `max_num_batched_tokens`、`max_num_seqs`、chunked prefill、decode 请求优先和 preemption 观测的公开说明：scheduler 必须同时看 token budget、running 数量、prefill / decode 平衡和 KV 容量。
3. vLLM / PagedAttention 论文对 block-based KV cache 的口径：continuous batching 不是单独的 batch 技巧，而是和 KV block allocation、free、reuse、admission control 一起成立。

本章不实现真实 vLLM scheduler、PagedAttention kernel、prefix cache、抢占恢复、CPU swap、多租户优先级、分布式 worker、异步 HTTP server、CUDA graph 或生产级 fairness。我们只保留最小可运行骨架：

```text
waiting queue -> running set -> BatchPlan -> BatchBuilder -> OutputProcessor -> cleanup / metrics
```

本章新增公式和 demo 只用于验证升级门禁：

```text
naive request-level batch lock 是否可见；
iteration boundary 是否被使用；
请求能否动态加入和动态退出；
decode-first 与 bounded prefill 是否同时成立；
token budget、max running requests、KV capacity 是否共同约束调度；
finished / cancelled 请求是否能释放资源。
```

## 48.1 本章目标

读完本章，你应该能讲清：

1. naive batching 和 continuous batching 的区别。
2. 为什么 request-level batch 不适合在线 LLM serving。
3. iteration-level batching 的核心思想。
4. prefill 和 decode 如何混排。
5. token budget、max running requests、KV capacity 如何共同约束调度。
6. continuous batching 如何影响 TTFT、TPOT、吞吐和显存使用。
7. 如何从一个简单 waiting/running scheduler 改造成 continuous batching scheduler。
8. 如何设计 scheduler 日志和压测指标验证改造效果。

## 48.2 naive scheduler 长什么样

最简单的 scheduler 通常是 request-level batching：

```text
收集一批请求 -> 一起 prefill -> 一起 decode 到全部完成 -> 再处理下一批请求
```

伪代码类似：

```python
def naive_generate(requests):
    batch = requests[:max_batch_size]

    prefill(batch)

    while not all_finished(batch):
        decode(batch)

    return outputs(batch)
```

这种设计适合离线批处理，但不适合在线服务。

在线服务中，请求是持续到达的：

```text
t=0ms    req A arrives
t=20ms   req B arrives
t=35ms   req C arrives
t=80ms   req D arrives
```

如果 req A 已经进入 decode，naive scheduler 可能会让 B/C/D 等到 A 完成后才能进来。

这会导致：

1. 新请求 TTFT 很差。
2. GPU batch size 可能很小。
3. 已完成请求不能及时释放 batch slot。
4. 长输出请求拖住整批请求。
5. 请求长度差异越大，浪费越明显。

这就是 request-level batch 的根本问题：

```text
batch 的生命周期被最长请求绑定。
```

## 48.3 一个直观例子

假设有三个请求：

| 请求 | prompt 长度 | output 长度 |
|---|---:|---:|
| A | 128 | 200 |
| B | 128 | 20 |
| C | 128 | 30 |

naive batching 会把 A/B/C 组成一批。

prefill 后开始 decode。

B 在第 20 步完成。

C 在第 30 步完成。

A 要到第 200 步才完成。

如果 batch slot 不能动态补新请求，那么第 31 到第 200 步，原本属于 B/C 的位置就浪费了。

更糟的是，如果 D/E/F 在这期间到达，它们可能需要等 A 完成后才能进入下一批。

于是系统会出现：

```text
GPU 还在跑，但 batch 利用率越来越低；新请求还在排队，TTFT 越来越差。
```

continuous batching 解决的正是这个问题。

## 48.4 continuous batching 的核心思想

continuous batching 的核心不是“把 batch 做大”。

它的核心是：

```text
每一轮 decode iteration 都重新调度，把新请求、未完成请求、刚释放资源的请求动态组合成当前最合适的 batch。
```

也可以叫 iteration-level batching。

对比一下：

| 方式 | batch 形成时机 | batch 生命周期 | 新请求何时进入 |
|---|---|---|---|
| naive batching | 一批请求开始时 | 直到整批完成 | 下一批开始时 |
| continuous batching | 每轮 iteration | 只对当前 step 有效 | 下一轮或若干轮后即可进入 |

continuous batching 的执行节奏是：

```text
step 0: 调度 A/B/C prefill
step 1: 调度 A/B/C decode
step 2: 调度 A/B/C decode，同时接收 D/E
step 3: 调度 A/B/C decode + D prefill chunk
step 4: B 完成，调度 A/C/D decode + E prefill chunk
step 5: C 完成，调度 A/D/E decode + F prefill chunk
```

注意这里的 batch 不是固定的一组请求。

每一轮 scheduler 都会重新做选择。

## 48.5 continuous batching 不等于无限插队

continuous batching 容易被误解成：

```text
新请求一来就马上插进当前 batch。
```

实际不是。

GPU forward 一旦开始，当前 batch 就固定了。

新请求最多只能进入下一轮或后续某一轮。

所以 continuous batching 的粒度是：

```text
iteration boundary
```

也就是每次 model forward 结束后，scheduler 有一次重新选择 batch 的机会。

在这个边界上，scheduler 会考虑：

1. 哪些 running 请求需要继续 decode。
2. 哪些 waiting 请求可以开始 prefill。
3. 是否有 KV 空间。
4. 是否还有 token budget。
5. 是否超过 max running requests。
6. 是否要优先 decode，避免 TPOT 抖动。
7. 是否允许 prefill chunk 混入本轮。

因此 continuous batching 的本质是：

```text
在 iteration boundary 做动态重组，而不是在 GPU forward 中途修改 batch。
```

## 48.6 第一步：定义调度输入和输出

升级 scheduler 前，先要明确接口。

Scheduler 的输入包括：

| 输入 | 含义 |
|---|---|
| `waiting` | 尚未开始或等待 prefill 的请求 |
| `running` | 已经占用 KV、正在 prefill/decode 的请求 |
| `finished` | 已完成、待清理或已清理的请求 |
| `kv_manager` | 查询和分配 KV 空间 |
| `token_budget` | 本轮最多计算多少 token |
| `max_running_requests` | 最多同时运行多少请求 |
| `policy` | decode-first、prefill-first、fairness 等策略 |

Scheduler 的输出不应该是 tensor。

它应该输出 batch plan：

```python
BatchPlan(
    scheduled=[
        ScheduleItem(req=A, kind="decode", token_start=128, token_count=1),
        ScheduleItem(req=B, kind="decode", token_start=256, token_count=1),
        ScheduleItem(req=C, kind="prefill", token_start=0, token_count=512),
    ],
    num_prefill_tokens=512,
    num_decode_tokens=2,
)
```

为什么不是直接返回 `input_ids`？

因为 `input_ids`、`positions`、`block_tables`、`slot_mapping` 应该由 `BatchBuilder` 构造。

Scheduler 只负责逻辑决策。

## 48.7 Request 需要增加哪些字段

为了支持 continuous batching，`Request` 需要记录更多执行进度。

最小字段包括：

| 字段 | 含义 |
|---|---|
| `prompt_tokens` | prompt token 列表 |
| `output_tokens` | 已生成 token 列表 |
| `computed_prompt_tokens` | prompt 中已经完成 prefill 的 token 数 |
| `prefill_done` | prompt 是否完成 prefill |
| `num_generated_tokens` | 已生成 token 数 |
| `status` | waiting/running/finished/aborted |
| `kv_blocks` | 已分配 KV block |
| `last_token` | decode 本轮要输入的 token |
| `arrival_time` | 请求到达时间 |
| `first_scheduled_time` | 第一次被调度时间 |
| `first_token_time` | 首 token 时间 |
| `last_decode_time` | 上次 decode 完成时间 |

`computed_prompt_tokens` 很关键。

没有 chunked prefill 时，它要么是 0，要么是完整 prompt 长度。

有 chunked prefill 后，它可以是：

```text
0 -> 512 -> 1024 -> 1536 -> prompt_len
```

它让 scheduler 知道：

```text
这个请求还剩多少 prompt token 没有计算？
```

## 48.8 最小 continuous batching 状态机

请求状态可以这样设计：

```text
WAITING
  |
  v
RUNNING_PREFILL
  |
  v
RUNNING_DECODE
  |
  +--> FINISHED
  +--> ABORTED
  +--> ERROR
```

如果支持 chunked prefill，`RUNNING_PREFILL` 会持续多轮。

状态转移如下：

| 当前状态 | 条件 | 下一状态 |
|---|---|---|
| `WAITING` | 被 scheduler 接纳，KV 分配成功 | `RUNNING_PREFILL` |
| `RUNNING_PREFILL` | prompt 未算完 | `RUNNING_PREFILL` |
| `RUNNING_PREFILL` | prompt 算完并采样首 token | `RUNNING_DECODE` |
| `RUNNING_DECODE` | 未达到停止条件 | `RUNNING_DECODE` |
| `RUNNING_DECODE` | EOS/max_tokens/stop | `FINISHED` |
| 任意运行状态 | 用户取消 | `ABORTED` |
| 任意运行状态 | OOM/异常 | `ERROR` |

Scheduler 只应该选择：

1. `WAITING` 中可以进入 prefill 的请求。
2. `RUNNING_PREFILL` 中还没算完 prompt 的请求。
3. `RUNNING_DECODE` 中还没完成的请求。

已经 `FINISHED`、`ABORTED`、`ERROR` 的请求不应该再被调度。

## 48.9 token budget 的作用

continuous batching 的关键约束是 token budget。

假设本轮 token budget 是 2048。

那么 scheduler 可以选择：

```text
方案 A：2048 个 prefill token
方案 B：1024 个 prefill token + 128 个 decode token
方案 C：512 个 prefill token + 256 个 decode token
方案 D：只做 512 个 decode token
```

注意 decode token 和 prefill token 的计算成本不完全等价。

但是在教学版 engine 中，可以先用 token 数作为近似预算。

token budget 至少解决三个问题：

1. 防止一轮 batch 太大导致 OOM。
2. 防止长 prompt prefill 独占 GPU 太久。
3. 控制一轮 step 时间，让 decode TPOT 更稳定。

如果没有 token budget，一个 32K prompt 可能会让所有 decode 请求等很久。

这会造成：

```text
平均吞吐看起来还行，但 TPOT p99 非常差。
```

面试里讲 continuous batching，一定要同时讲 token budget。

否则回答会停留在“动态组 batch”的表面。

## 48.10 decode-first 策略

很多推理框架会优先保证 decode 请求。

原因是 decode 阶段直接影响用户感知的流式输出速度。

如果用户已经看到模型开始输出，但后续 token 一卡一卡的，体验会很差。

decode-first 的思想是：

```text
每一轮先给 running decode 请求分配 1 token，再用剩余 token budget 处理 prefill。
```

伪代码：

```python
def schedule_decode_first(waiting, running, token_budget):
    plan = []

    for req in running:
        if token_budget <= 0:
            break
        if req.status == "RUNNING_DECODE" and not req.finished:
            plan.append(ScheduleItem(req, "decode", req.next_token_pos, 1))
            token_budget -= 1

    while waiting and token_budget > 0:
        req = waiting[0]
        chunk = min(req.remaining_prompt_tokens, token_budget)
        plan.append(ScheduleItem(req, "prefill", req.computed_prompt_tokens, chunk))
        token_budget -= chunk
        waiting.pop(0)
        running.append(req)

    return plan
```

decode-first 的优点：

1. TPOT 更稳定。
2. 流式输出更平滑。
3. 已经开始输出的请求不容易被长 prompt 饿死。

缺点：

1. waiting 请求的 TTFT 可能变差。
2. 如果 running decode 很多，prefill 可能长期进不来。
3. 需要 admission control 防止 running 请求过多。

所以 decode-first 不能单独使用，还要配合 max running requests、prefill quota 或 fairness 策略。

## 48.11 prefill-first 策略的问题

prefill-first 看起来也有道理：

```text
尽快让新请求完成 prefill，尽快产生首 token。
```

这有利于 TTFT。

但是如果 prefill 长度很大，它会拖慢所有 running decode 请求。

例如：

```text
running: 128 个 decode 请求，每个需要 1 token
waiting: 1 个 8192-token prompt
token_budget: 8192
```

如果本轮全部拿去做 prefill，128 个 decode 请求会等完整个长 prompt forward。

用户感知就是：

```text
已经在输出的请求突然卡住。
```

所以在线服务一般不会简单 prefill-first。

更常见的是：

```text
decode-first + bounded prefill
```

或者：

```text
decode 优先，但每轮保留一部分 budget 给 prefill，避免新请求饿死。
```

## 48.12 bounded prefill

bounded prefill 的意思是限制每轮 prefill token 数。

例如：

```text
total_token_budget = 4096
max_prefill_tokens_per_step = 1024
```

那么每轮最多只处理 1024 个 prefill token，剩余 budget 留给 decode 或其他请求。

如果一个 prompt 有 8192 token，就会被拆成 8 轮：

```text
step 1: prefill [0:1024]
step 2: prefill [1024:2048]
...
step 8: prefill [7168:8192]
```

这就是 chunked prefill 的基础。

bounded prefill 解决的是：

1. 长 prompt 阻塞 decode。
2. 单轮 forward 时间过长。
3. TPOT p99 抖动。
4. GPU memory 峰值不可控。

代价是：

1. 长 prompt 的 TTFT 可能增加。
2. scheduler 和 BatchBuilder 更复杂。
3. 需要维护 `computed_prompt_tokens`。

所以 bounded prefill 是吞吐、TTFT、TPOT 之间的权衡。

## 48.13 admission control：不是所有 waiting 都能进来

continuous batching 还有一个容易忽略的问题：

```text
如果每轮都不断接纳新请求，running 队列会越来越大。
```

running 请求越多，KV cache 占用越高。

decode 每轮要处理的 token 也越多。

如果 running 太大，会出现：

1. KV cache 被占满。
2. decode batch 太大，单步延迟上升。
3. 新请求虽然进来了，但整体 TPOT 变差。
4. OOM 风险增加。

所以 scheduler 需要 admission control。

最简单的规则：

```text
len(running) < max_running_requests
```

更真实的规则：

```text
KV cache 是否足够容纳 prompt + max_new_tokens 的上界？
```

或者：

```text
当前 running decode token 数是否超过上限？
```

在教学版 engine 中，可以先用三个约束：

1. `max_running_requests`。
2. `token_budget`。
3. `kv_manager.can_allocate(req)`。

这三个约束已经足够解释 continuous batching 的核心行为。

## 48.14 一个更完整的调度伪代码

下面是一个教学版 continuous batching scheduler。

它采用：

1. decode-first。
2. bounded prefill。
3. max running requests。
4. KV capacity check。

```python
def schedule(
    waiting,
    running,
    kv_manager,
    token_budget,
    max_prefill_tokens,
    max_running_requests,
):
    plan = []
    remaining_budget = token_budget
    remaining_prefill_budget = max_prefill_tokens

    # 1. Prioritize one decode token for each running decode request.
    for req in list(running):
        if remaining_budget <= 0:
            break
        if req.status != "RUNNING_DECODE":
            continue
        if req.finished or req.aborted:
            continue

        plan.append(ScheduleItem(req=req, kind="decode", token_start=req.next_pos, token_count=1))
        remaining_budget -= 1

    # 2. Continue chunked prefill for already admitted requests.
    for req in list(running):
        if remaining_budget <= 0 or remaining_prefill_budget <= 0:
            break
        if req.status != "RUNNING_PREFILL":
            continue

        chunk = min(req.remaining_prompt_tokens, remaining_budget, remaining_prefill_budget)
        if chunk <= 0:
            continue

        plan.append(ScheduleItem(req=req, kind="prefill", token_start=req.computed_prompt_tokens, token_count=chunk))
        remaining_budget -= chunk
        remaining_prefill_budget -= chunk

    # 3. Admit new waiting requests if there is room.
    while waiting and remaining_budget > 0 and remaining_prefill_budget > 0:
        if len(running) >= max_running_requests:
            break

        req = waiting[0]
        if not kv_manager.can_allocate(req):
            break

        waiting.pop(0)
        running.append(req)
        req.status = "RUNNING_PREFILL"
        kv_manager.allocate(req)

        chunk = min(req.remaining_prompt_tokens, remaining_budget, remaining_prefill_budget)
        if chunk <= 0:
            break

        plan.append(ScheduleItem(req=req, kind="prefill", token_start=0, token_count=chunk))
        remaining_budget -= chunk
        remaining_prefill_budget -= chunk

    return BatchPlan(plan)
```

这段代码不是生产实现，但结构是对的。

它体现了几个关键原则：

1. decode 请求每轮最多调度 1 token。
2. prefill 可以被切 chunk。
3. running prefill 比新 waiting 请求优先，避免已经接纳的请求长期卡在 prefill 中间。
4. 新请求进入 running 前先检查 KV capacity。
5. batch plan 只描述逻辑 token span，不构造 tensor。

## 48.15 调度顺序的权衡

上面的顺序是：

```text
decode -> existing prefill -> new prefill
```

这是一个稳妥的教学版策略。

但不同系统可能选择不同顺序。

| 顺序 | 优点 | 风险 |
|---|---|---|
| decode -> prefill | TPOT 稳定 | TTFT 可能变差 |
| prefill -> decode | TTFT 可能更好 | decode 抖动大 |
| existing prefill -> new prefill -> decode | 已接纳请求更快首 token | streaming 卡顿 |
| decode + fixed prefill quota | 折中 | 参数需要调优 |
| priority-based | 可支持 SLA | 实现复杂 |

面试时不要说某一种策略永远最好。

正确表达是：

```text
调度策略是在 TTFT、TPOT、吞吐、显存和公平性之间做权衡。
```

## 48.16 BatchBuilder 需要怎么变

continuous batching 升级后，BatchBuilder 的输入会更复杂。

原来可能是：

```text
一批请求全部 prefill
```

或者：

```text
一批请求全部 decode
```

现在可能是混合 batch：

```text
req A: decode 1 token
req B: decode 1 token
req C: prefill tokens [0:512]
req D: prefill tokens [1024:1536]
```

BatchBuilder 要为每个 `ScheduleItem` 生成：

| metadata | 说明 |
|---|---|
| `input_ids` | 本轮要计算的 token |
| `positions` | 每个 token 的 position |
| `seq_lens` | 当前序列长度 |
| `slot_mapping` | 每个 token 写入 KV 的位置 |
| `block_tables` | 每个请求的 KV block 表 |
| `query_lens` | 每个请求本轮 query token 数 |
| `logits_indices` | 哪些 token 位置需要取 logits |

对于 decode：

```text
input_ids = [last_generated_token]
positions = [current_seq_len - 1]
query_len = 1
```

对于 prefill chunk：

```text
input_ids = prompt_tokens[start:start+chunk]
positions = [start, start+1, ...]
query_len = chunk
```

如果 prefill chunk 不是最后一段，通常不需要采样输出 token。

只有当 prompt prefill 完成后，才需要取最后一个 prompt token 的 logits 采样首 token。

所以 BatchBuilder 还要告诉 OutputProcessor：

```text
哪些 request 在本轮 forward 后需要采样？
```

这就是 `logits_indices` 的作用。

## 48.17 OutputProcessor 需要怎么变

continuous batching 后，OutputProcessor 不能再假设每个请求每轮都有输出 token。

对于 prefill chunk，有两种情况：

1. 不是最后一个 chunk：只更新 `computed_prompt_tokens`，不采样。
2. 是最后一个 chunk：更新 `computed_prompt_tokens`，采样首 token，进入 decode。

对于 decode：

1. append 新 token。
2. 更新 `num_generated_tokens`。
3. 检查 EOS/stop/max_tokens。
4. 更新 `last_decode_time`。
5. 如果完成，标记 `FINISHED`。

伪代码：

```python
def process_outputs(batch_plan, sampled_tokens):
    for item in batch_plan.items:
        req = item.req

        if item.kind == "prefill":
            req.computed_prompt_tokens += item.token_count
            if req.computed_prompt_tokens < len(req.prompt_tokens):
                continue

            token = sampled_tokens[req.request_id]
            req.output_tokens.append(token)
            req.first_token_time = now()
            req.status = "RUNNING_DECODE"
            continue

        if item.kind == "decode":
            token = sampled_tokens[req.request_id]
            req.output_tokens.append(token)
            req.last_decode_time = now()
            if should_stop(req):
                req.status = "FINISHED"
```

这里的关键点是：

```text
被调度不等于一定产生输出 token。
```

prefill 中间 chunk 只是在补 KV。

## 48.18 KV capacity 如何影响调度

continuous batching 会让多个请求长期同时处于 running 状态。

这意味着 KV cache 是 admission control 的核心约束。

一个请求的 KV 需求大致是：

```text
prompt_len + max_new_tokens
```

如果只按 prompt_len 分配，decode 过程中可能不断追加 KV，最终 OOM。

教学版可以先简化成两种策略。

策略一：保守预分配。

```text
进入 running 前，为 prompt_len + max_new_tokens 预留 KV。
```

优点：不会 decode 到一半没空间。

缺点：显存利用率低，因为很多请求不会生成到 max_new_tokens。

策略二：按需追加。

```text
prefill 时分配 prompt KV，decode 每步追加 1 token KV。
```

优点：显存利用率高。

缺点：decode 中途可能 KV 不足，需要抢占、等待或终止。

教学版第一阶段建议用保守预分配，逻辑简单。

等第 49 章升级 paged KV cache 后，再讨论更细的 block 分配、回收和抢占。

## 48.19 抢占和暂停：第一版可以先不做

生产级 scheduler 可能会支持 preemption。

例如：

```text
KV cache 不够时，把低优先级请求暂停，释放部分 KV，优先服务高优先级请求。
```

或者：

```text
把某个请求的 KV swap 到 CPU，再恢复。
```

这些能力很复杂，第一版不建议实现。

第一版可以采用简单规则：

1. 如果 KV 不够，新请求继续留在 waiting。
2. 已经 running 的请求不抢占。
3. 如果 decode 追加 KV 失败，标记 error 或等待下一轮。
4. 通过 max running requests 降低 KV 打满概率。

这样能保持 scheduler 清晰。

等你理解 continuous batching 后，再考虑 preemption。

## 48.20 scheduler debug log

continuous batching 一定要加 debug log。

否则出了问题很难定位。

每轮至少打印：

| 字段 | 含义 |
|---|---|
| `step_id` | 第几轮 engine step |
| `waiting_count` | waiting 请求数 |
| `running_count` | running 请求数 |
| `scheduled_count` | 本轮调度请求数 |
| `prefill_reqs` | 本轮 prefill 请求数 |
| `decode_reqs` | 本轮 decode 请求数 |
| `prefill_tokens` | 本轮 prefill token 数 |
| `decode_tokens` | 本轮 decode token 数 |
| `token_budget_used` | 使用了多少 token budget |
| `kv_free_blocks` | 剩余 KV block |
| `finished_count` | 本轮完成请求数 |

示例日志：

```text
step=128 waiting=42 running=96 scheduled=96 prefill_reqs=4 decode_reqs=92 prefill_tokens=1024 decode_tokens=92 budget=1116/2048 kv_free=381 finished=3
```

这行日志可以快速回答：

1. scheduler 是否在持续接纳新请求。
2. decode 是否被长 prefill 饿死。
3. token budget 是否利用充分。
4. KV 是否成为瓶颈。
5. running 是否接近 max_running_requests。

## 48.21 压测场景设计

验证 continuous batching，不能只跑几个 prompt。

至少要设计几类 workload。

场景一：短 prompt、短输出。

```text
prompt_len: 64-128
output_len: 16-32
arrival: fixed QPS
```

观察重点：

1. QPS。
2. TTFT。
3. batch 利用率。

场景二：短 prompt、长输出。

```text
prompt_len: 64-128
output_len: 256-512
arrival: fixed QPS
```

观察重点：

1. running 队列长度。
2. decode batch size。
3. TPOT p99。

场景三：长 prompt、短输出。

```text
prompt_len: 4096-8192
output_len: 16-32
arrival: fixed QPS
```

观察重点：

1. TTFT。
2. decode 是否被 prefill 阻塞。
3. chunked prefill 的效果。

场景四：混合长度。

```text
80% short prompt + 20% long prompt
output_len: mixed
arrival: poisson
```

观察重点：

1. p99 TTFT。
2. p99 TPOT。
3. 短请求是否被长请求拖慢。
4. token budget 参数是否合理。

场景五：突发流量。

```text
前 10 秒低 QPS，中间 10 秒高 QPS，后 10 秒恢复低 QPS
```

观察重点：

1. waiting 队列积压。
2. 恢复时间。
3. KV cache 使用峰值。
4. abort/error 数量。

## 48.22 对比指标

升级前后要对比 naive scheduler 和 continuous batching。

至少记录：

| 指标 | 预期变化 |
|---|---|
| output tokens/s | 通常上升 |
| QPS | 通常上升 |
| 平均 TTFT | 取决于策略 |
| p99 TTFT | 可能改善，也可能因 decode-first 变差 |
| 平均 TPOT | 通常更稳定 |
| p99 TPOT | 通常改善 |
| GPU utilization | 通常上升 |
| KV usage | 通常上升 |
| waiting queue length | 高负载下更平滑 |
| OOM 次数 | 取决于 admission control |

不要只看平均值。

推理 serving 的问题经常藏在 p99。

尤其要看：

```text
TTFT p99 和 TPOT p99
```

如果 continuous batching 提升了吞吐，但 TPOT p99 爆炸，用户体验可能更差。

## 48.23 常见 bug

bug 一：prefill 中间 chunk 也采样 token。

```text
结果：prompt 还没算完就开始生成，position 和 KV 状态错乱。
```

bug 二：decode 请求一轮被调度多个 token。

```text
结果：没有 speculative decoding 的情况下，下一 token 依赖上一 token，不能一次调度多个未知 token。
```

bug 三：请求 finished 后没有释放 KV。

```text
结果：KV usage 持续上涨，最终新请求进不来。
```

bug 四：waiting 请求进入 running 后，KV 分配失败但状态已修改。

```text
结果：请求既不在 waiting，也不能正常 running，变成悬挂状态。
```

bug 五：token budget 只统计 prefill，不统计 decode。

```text
结果：running decode 很多时，实际 batch 远超预算。
```

bug 六：BatchBuilder 没区分 logits 位置。

```text
结果：从 prefill chunk 中间位置采样，或者 decode logits 对错请求。
```

bug 七：没有处理 abort。

```text
结果：客户端断开后请求还在占用 KV 和 decode slot。
```

bug 八：running prefill 被新 waiting 请求长期挤压。

```text
结果：某些请求进入 running 后迟迟拿不到首 token。
```

## 48.24 面试高频问题

问题一：continuous batching 和 naive batching 的区别是什么？

回答要点：naive batching 在 request 级别形成固定 batch，一批请求通常要一起 decode 到完成，新请求要等下一批；continuous batching 在 iteration 级别动态重组 batch，每轮 forward 结束后可以加入新请求、移除完成请求，让 batch slot 持续被利用。

问题二：为什么 continuous batching 能提升吞吐？

回答要点：它避免 batch 生命周期被最长请求绑定，完成请求释放的 slot 可以被新请求填充；在线请求持续到达时，GPU 更容易保持较高 batch 利用率，减少小 batch 和空转。

问题三：continuous batching 如何影响 TTFT 和 TPOT？

回答要点：它通常提升吞吐并改善 TPOT 稳定性，但具体取决于策略。decode-first 可以降低 TPOT 抖动，但可能让 waiting 请求 TTFT 变差；prefill quota 和 chunked prefill 可以在 TTFT 和 TPOT 之间做权衡。

问题四：为什么需要 token budget？

回答要点：token budget 限制每轮处理的 token 数，防止长 prompt prefill 独占 GPU 或导致 OOM，也让每轮 step 时间更可控。没有 token budget，长 prefill 会严重影响 decode TPOT p99。

问题五：decode-first 有什么优缺点？

回答要点：优点是保障已经进入 decode 的请求每轮持续产生 token，TPOT 和 streaming 体验更稳定；缺点是如果 running decode 太多，waiting 请求可能迟迟无法 prefill，TTFT 变差，所以需要 max running requests、prefill quota 或 fairness 策略。

问题六：continuous batching 下 BatchBuilder 为什么更复杂？

回答要点：batch 中可能同时有 decode、完整 prefill、prefill chunk。BatchBuilder 需要为不同请求生成 input ids、positions、seq lens、slot mapping、block table、query lens 和 logits indices，并区分哪些请求本轮需要采样。

## 48.25 标准回答模板

如果面试官问“你如何把 naive scheduler 升级成 continuous batching”，可以这样回答：

```text
我会先把 scheduler 的输出定义成 batch plan，而不是直接构造 tensor。每个 ScheduleItem 描述一个 request 本轮执行 prefill 还是 decode、从哪个 token 位置开始、执行多少 token。BatchBuilder 再把这个逻辑计划转成 input_ids、positions、block table、slot mapping 和 logits indices。

naive scheduler 的问题是 request-level batch 生命周期被最长请求绑定，一批请求里短请求完成后 slot 不能及时复用，新请求要等整批完成，在线服务下会导致 GPU 利用率低、TTFT 差。continuous batching 改成 iteration-level batching，每轮 model forward 结束后重新调度：移除完成请求，继续调度 running decode，请求允许在 iteration boundary 加入 running。

具体策略上，我会先实现 decode-first + bounded prefill。每轮先给 RUNNING_DECODE 请求调度 1 个 token，保证 TPOT 和 streaming 稳定；然后给 RUNNING_PREFILL 请求继续执行 prompt chunk；最后在 token budget、max prefill tokens、max running requests 和 KV capacity 允许时，从 waiting 队列接纳新请求。长 prompt 会通过 computed_prompt_tokens 被切成多个 prefill chunk，避免单个长 prompt 阻塞所有 decode。

验证时我会对比 naive scheduler 和 continuous batching 在不同 workload 下的 TTFT、TPOT、output tokens/s、QPS、KV usage 和 waiting queue length，尤其看 p99 TTFT/TPOT。还会加 scheduler decision log，记录每轮 prefill/decode 请求数、token 数、budget 使用和 KV 剩余量，用于定位长 prompt 阻塞、running 过多或 KV 不足问题。
```

## 48.26 Naive Scheduler 到 Continuous Batching 升级门禁和可运行 demo

把 naive scheduler 升级成 continuous batching，不能只写一句“每轮动态组 batch”。最小验收要能同时量化静态 batch 空洞、每轮 token budget、prefill chunk、running 上限、KV 准入和 cleanup。

先定义请求 `i` 的实际输出 token 数为 `y_i`。如果 request-level batch 被最长输出锁住，静态 decode 行数近似为：

```math
W_{\mathrm{static}}=N\max_i y_i
```

continuous batching 的 decode 行数等于每轮真正参与 decode 的请求数之和：

```math
W_{\mathrm{cont}}=\sum_{t=1}^{T}|D_t|
```

本轮 token budget 约束：

```math
|D_t|+\sum_{i\in P_t}c_{i,t}\le B_{\mathrm{tok}}
```

其中 `D_t` 是本轮 decode 请求集合，`P_t` 是本轮 prefill 请求集合，`c_{i,t}` 是请求 `i` 本轮 prefill chunk token 数。

bounded prefill 约束：

```math
\sum_{i\in P_t}c_{i,t}\le B_{\mathrm{prefill}}
```

running 上限和 KV 准入约束：

```math
R_t\le R_{\max}
```

```math
K_t+K_i^{\mathrm{reserve}}\le K_{\max}
```

最终升级门禁可以写成：

```math
G_{\mathrm{upgrade}}=G_{\mathrm{lock}}G_{\mathrm{iter}}G_{\mathrm{join}}G_{\mathrm{exit}}G_{\mathrm{decode}}G_{\mathrm{prefill}}G_{\mathrm{kv}}G_{\mathrm{cleanup}}
```

下面这个 0 依赖 demo 不跑真实模型，只模拟 scheduler 的核心状态变化：A/B/C 是第一批请求，B/C 比 A 短；X 会在 waiting 中被取消；D/E 在 A 还未结束时动态加入；E 的长 prompt 被 bounded prefill 切成多轮；X 先被 `max_running_requests` 和 `kv_capacity` 延迟，取消后必须释放队头阻塞。

```python
from dataclasses import dataclass
import math


@dataclass
class Request:
    request_id: str
    arrival_step: int
    prompt_len: int
    max_new_tokens: int
    target_new_tokens: int
    cancel_step: int = None
    status: str = "NEW"
    computed_prompt_tokens: int = 0
    generated_tokens: int = 0
    first_scheduled_step: int = None
    first_token_step: int = None
    finish_step: int = None

    @property
    def remaining_prompt_tokens(self):
        return self.prompt_len - self.computed_prompt_tokens


@dataclass
class ScheduleItem:
    request_id: str
    kind: str
    token_start: int
    token_count: int


class ToyKVManager:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.allocations = {}
        self.max_used_blocks = 0

    def reserve_blocks(self, req):
        return math.ceil((req.prompt_len + req.max_new_tokens) / self.block_size)

    def used_blocks(self):
        return sum(self.allocations.values())

    def free_blocks(self):
        return self.total_blocks - self.used_blocks()

    def can_reserve(self, req):
        return self.reserve_blocks(req) <= self.free_blocks()

    def reserve(self, req):
        blocks = self.reserve_blocks(req)
        self.allocations[req.request_id] = blocks
        self.max_used_blocks = max(self.max_used_blocks, self.used_blocks())

    def release(self, req):
        self.allocations.pop(req.request_id, None)


class ToyNaiveToContinuousBatcher:
    def __init__(self, requests, token_budget, max_prefill_tokens, max_running_requests, kv_manager):
        self.requests = {req.request_id: req for req in requests}
        self.token_budget = token_budget
        self.max_prefill_tokens = max_prefill_tokens
        self.max_running_requests = max_running_requests
        self.kv_manager = kv_manager
        self.waiting = []
        self.running = []
        self.finished_order = []
        self.cancelled = []
        self.deferred_reasons = []
        self.trace = []

    def defer(self, req, reason):
        item = f"{req.request_id}:{reason}"
        if item not in self.deferred_reasons:
            self.deferred_reasons.append(item)

    def arrive_and_cancel(self, step):
        for req in self.requests.values():
            if req.arrival_step == step:
                req.status = "WAITING"
                self.waiting.append(req)

        for req in list(self.waiting) + list(self.running):
            if req.cancel_step == step:
                req.status = "ABORTED"
                if req in self.waiting:
                    self.waiting.remove(req)
                if req in self.running:
                    self.running.remove(req)
                    self.kv_manager.release(req)
                self.cancelled.append(req.request_id)

    def schedule_step(self, step):
        remaining_budget = self.token_budget
        remaining_prefill = self.max_prefill_tokens
        plan = []

        for req in list(self.running):
            if remaining_budget <= 0:
                break
            if req.status == "RUNNING_DECODE":
                plan.append(ScheduleItem(req.request_id, "decode", req.prompt_len + req.generated_tokens - 1, 1))
                remaining_budget -= 1

        for req in list(self.running):
            if remaining_budget <= 0 or remaining_prefill <= 0:
                break
            if req.status != "RUNNING_PREFILL":
                continue
            chunk = min(req.remaining_prompt_tokens, remaining_budget, remaining_prefill)
            if chunk < req.remaining_prompt_tokens:
                self.defer(req, "prefill_budget")
            plan.append(ScheduleItem(req.request_id, "prefill", req.computed_prompt_tokens, chunk))
            remaining_budget -= chunk
            remaining_prefill -= chunk

        while self.waiting and remaining_budget > 0 and remaining_prefill > 0:
            if len(self.running) >= self.max_running_requests:
                self.defer(self.waiting[0], "max_running_requests")
                break
            req = self.waiting[0]
            if not self.kv_manager.can_reserve(req):
                self.defer(req, "kv_capacity")
                break

            self.waiting.pop(0)
            self.running.append(req)
            req.status = "RUNNING_PREFILL"
            req.first_scheduled_step = step
            self.kv_manager.reserve(req)

            chunk = min(req.remaining_prompt_tokens, remaining_budget, remaining_prefill)
            if chunk < req.remaining_prompt_tokens:
                self.defer(req, "prefill_budget")
            plan.append(ScheduleItem(req.request_id, "prefill", req.computed_prompt_tokens, chunk))
            remaining_budget -= chunk
            remaining_prefill -= chunk

        return plan

    def execute(self, step, plan):
        for item in plan:
            req = self.requests[item.request_id]
            if item.kind == "prefill":
                req.computed_prompt_tokens += item.token_count
                if req.computed_prompt_tokens < req.prompt_len:
                    continue
                req.generated_tokens += 1
                req.status = "RUNNING_DECODE"
                if req.first_token_step is None:
                    req.first_token_step = step + 1
            else:
                req.generated_tokens += 1

            if req.generated_tokens >= req.target_new_tokens:
                req.status = "FINISHED"
                req.finish_step = step + 1
                self.finished_order.append(req.request_id)

        for req in list(self.running):
            if req.status == "FINISHED":
                self.running.remove(req)
                self.kv_manager.release(req)

    def run(self, max_steps=20):
        for step in range(max_steps):
            self.arrive_and_cancel(step)
            plan = self.schedule_step(step)
            self.trace.append(
                {
                    "step": step,
                    "plan_kinds": [item.kind for item in plan],
                    "decode": [item.request_id for item in plan if item.kind == "decode"],
                    "prefill": [(item.request_id, item.token_start, item.token_count) for item in plan if item.kind == "prefill"],
                    "waiting": [req.request_id for req in self.waiting],
                    "running": [req.request_id for req in self.running],
                    "used_blocks": self.kv_manager.used_blocks(),
                    "free_blocks": self.kv_manager.free_blocks(),
                }
            )
            self.execute(step, plan)
            live = self.waiting or self.running
            future = any(req.status == "NEW" for req in self.requests.values())
            if not live and not future:
                break

        completed = [req for req in self.requests.values() if req.status == "FINISHED"]
        static_rows = len(completed) * max(req.target_new_tokens for req in completed)
        continuous_rows = sum(req.target_new_tokens for req in completed)
        queue_wait = {
            req.request_id: req.first_scheduled_step - req.arrival_step
            for req in completed
        }
        ttft = {
            req.request_id: req.first_token_step - req.arrival_step
            for req in completed
        }
        gates = {
            "request_level_lock_visible": static_rows > continuous_rows,
            "iteration_boundary_used": len(self.trace) > 1,
            "dynamic_join_visible": self.requests["D"].first_scheduled_step < self.requests["A"].finish_step,
            "dynamic_exit_visible": self.requests["B"].finish_step < self.requests["A"].finish_step,
            "decode_first_ready": all(
                "prefill" not in row["plan_kinds"][: row["plan_kinds"].count("decode")]
                for row in self.trace
            ),
            "bounded_prefill_ready": any("prefill_budget" in item for item in self.deferred_reasons),
            "kv_admission_ready": any("kv_capacity" in item for item in self.deferred_reasons),
            "cleanup_ready": self.kv_manager.used_blocks() == 0,
        }
        gates["scheduler_upgrade_gate"] = all(gates.values())
        summary = {
            "finished_order": self.finished_order,
            "cancelled": self.cancelled,
            "queue_wait_steps": queue_wait,
            "ttft_steps": ttft,
            "static_decode_rows": static_rows,
            "continuous_output_rows": continuous_rows,
            "saved_rows": static_rows - continuous_rows,
            "max_running": max(len(row["running"]) for row in self.trace),
            "max_kv_blocks_used": self.kv_manager.max_used_blocks,
            "deferred_reasons": self.deferred_reasons,
            "trace_tail": self.trace[-4:],
        }
        return summary, gates


requests = [
    Request("A", arrival_step=0, prompt_len=4, max_new_tokens=6, target_new_tokens=6),
    Request("B", arrival_step=0, prompt_len=4, max_new_tokens=2, target_new_tokens=2),
    Request("C", arrival_step=0, prompt_len=4, max_new_tokens=3, target_new_tokens=3),
    Request("X", arrival_step=1, prompt_len=12, max_new_tokens=4, target_new_tokens=4, cancel_step=4),
    Request("D", arrival_step=2, prompt_len=5, max_new_tokens=2, target_new_tokens=2),
    Request("E", arrival_step=3, prompt_len=8, max_new_tokens=2, target_new_tokens=2),
]

batcher = ToyNaiveToContinuousBatcher(
    requests=requests,
    token_budget=8,
    max_prefill_tokens=8,
    max_running_requests=3,
    kv_manager=ToyKVManager(total_blocks=8, block_size=4),
)
summary, gates = batcher.run()
print("scheduler_upgrade_summary=", summary)
print("scheduler_upgrade_gates=", gates)
```

一次运行的核心输出类似：

```text
scheduler_upgrade_summary= {'finished_order': ['B', 'C', 'A', 'D', 'E'], 'cancelled': ['X'], 'queue_wait_steps': {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 1}, 'ttft_steps': {'A': 1, 'B': 1, 'C': 2, 'D': 3, 'E': 3}, 'static_decode_rows': 30, 'continuous_output_rows': 15, 'saved_rows': 15, 'max_running': 3, 'max_kv_blocks_used': 8, 'deferred_reasons': ['X:max_running_requests', 'X:kv_capacity', 'E:prefill_budget'], 'trace_tail': [...]}
scheduler_upgrade_gates= {'request_level_lock_visible': True, 'iteration_boundary_used': True, 'dynamic_join_visible': True, 'dynamic_exit_visible': True, 'decode_first_ready': True, 'bounded_prefill_ready': True, 'kv_admission_ready': True, 'cleanup_ready': True, 'scheduler_upgrade_gate': True}
```

这个 demo 的重点不是性能数值，而是升级验收证据：

1. `static_decode_rows` 大于 `continuous_output_rows`，说明 request-level batch lock 的空洞可见。
2. D/E 在 A 还没完成时进入 running，说明 dynamic join 生效。
3. B/C 在 A 之前完成并释放 KV，说明 dynamic exit 和 cleanup 生效。
4. X 因 running 上限和 KV 容量被延迟，随后取消，说明 waiting 队列不能只入不出。
5. E 的 prompt 被切成多轮 prefill，说明 bounded prefill 没有让长 prompt 独占 iteration。
6. 最后 `scheduler_upgrade_gate=True`，说明调度、预算、准入和清理路径同时闭环。

## 48.27 小练习

1. 画出 naive batching 和 continuous batching 的时序图。
2. 构造 A/B/C 三个不同 output 长度请求，说明 naive batching 如何浪费 batch slot。
3. 为 `Request` 增加 `computed_prompt_tokens`、`status`、`first_token_time` 字段。
4. 实现一个返回 `BatchPlan` 的 scheduler，而不是直接返回 tensor。
5. 实现 decode-first 调度策略。
6. 增加 `max_prefill_tokens_per_step`，支持 bounded prefill。
7. 设计 `max_running_requests`，观察它对 TTFT 和 TPOT 的影响。
8. 给 scheduler 增加每轮 debug log。
9. 设计短 prompt 长输出 workload，对比 naive 和 continuous batching 的 output TPS。
10. 设计长 prompt 短输出 workload，观察 chunked prefill 对 TPOT p99 的影响。
11. 统计 waiting queue length 和 KV usage，判断瓶颈来自哪里。
12. 写一个面试回答：continuous batching 为什么不是简单把 batch size 调大？

## 48.28 本章总结

continuous batching 是 LLM serving scheduler 的核心能力。

naive batching 在 request 级别形成固定 batch，batch 生命周期被最长请求绑定，不适合持续请求到达的在线服务。

continuous batching 把调度粒度降到 iteration 级别，每轮 forward 结束后重新组合 batch，让完成请求释放资源，让新请求在合适时机进入 running，从而提升 batch 利用率和吞吐。

实现 continuous batching 时，关键不是一句“动态 batching”，而是要处理好 token budget、decode-first、bounded prefill、admission control、KV capacity、BatchBuilder metadata 和 OutputProcessor 状态更新。

调度策略没有绝对最优。decode-first 更重视 TPOT 和 streaming 平滑度，prefill quota 会影响 TTFT，max running requests 和 KV capacity 决定系统能接纳多少并发请求。

真正能证明升级有效的，是 scheduler log 和压测指标，尤其是 TTFT p99、TPOT p99、output TPS、KV usage 和 waiting queue length。

下一章会进入第 49 章：从 list KV cache 升级到 paged KV cache。
