# 第 51 章 实现 preemption、recompute 和 swap

上一章我们在 paged KV cache 之上实现了 prefix cache。

prefix cache 带来的收益很直接：复用已经算过的前缀 KV，降低重复 prefill 成本，改善 TTFT。

但它也带来了一个更现实的问题：

```text
KV blocks 更容易不够用。
```

即使不开 prefix cache，只要在线流量里存在长 prompt、长输出、高并发和突发请求，paged KV cache 也会遇到资源紧张。

这时 scheduler 不能只有两个选择：

```text
要么接请求，要么 OOM。
```

它需要第三种能力：

```text
在 KV 空间不足时，临时让出一部分 running 请求占用的资源，等资源恢复后再继续执行。
```

这就是本章要实现的 preemption。

## 51.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲教学版 paged KV cache 在 KV block 压力下的 preemption、recompute 和 swap 状态机。

公开资料校准主要参考三类口径：

1. vLLM optimization 文档对 KV cache 空间不足、preemption、recompute / swap 策略、`max_num_seqs`、`max_num_batched_tokens` 和 KV cache usage 的公开说明。
2. vLLM / PagedAttention 论文对 block-based KV cache 和动态 serving request 的资源压力背景。
3. 本书第 49、50 章对 paged KV cache、prefix cache、cached-free block、ref count 和 eviction 的教学抽象。

本章不实现真实异步 CPU/GPU KV copy、pinned memory、PCIe/NVLink 带宽模型、多 GPU / TP rank 一致 swap、KV connector、远程 KV cache、生产级优先级队列或完整 vLLM preemption policy。我们只验证一个最小闭环：

```text
KV budget 不足 -> 先 cache eviction -> 再 preempt running request -> recompute 或 swap 恢复 -> 记录指标
```

第二轮新增 demo 的验收重点是：

```text
preemption 和 cache eviction、abort、backpressure 是否区分清楚；
资源紧张时是否先淘汰 cached-free blocks，再 preempt active request；
recompute preemption 是否保留 prompt tokens 和 output tokens；
恢复时是否用 prompt + output 重建 KV；
swap out / swap in 是否保持逻辑 block 顺序；
swap in 失败时 CPU KV 是否保留；
preemption 指标是否能解释尾延迟风险。
```

## 51.1 本章目标

读完本章，你应该能讲清：

1. preemption 在推理框架里解决什么问题。
2. preemption、eviction、abort、backpressure 的区别。
3. recompute 和 swap 两种策略的核心差异。
4. 为什么 vLLM-like engine 通常更偏向 recompute。
5. scheduler 在什么时机触发 preemption。
6. 被 preempt 的 request 状态如何保存。
7. recompute 恢复时如何重新分配 KV blocks。
8. swap 恢复时如何从 CPU KV 拷回 GPU。
9. preemption 会怎样影响 TTFT、TPOT、吞吐和显存。
10. 如何用日志、指标和压测判断 preemption 是否过于频繁。

面试中要先说结论：

```text
preemption 是鲁棒性机制，不是性能优化手段。
```

它的目标是避免系统在 KV cache 压力下直接 OOM，同时让服务继续前进。

如果系统频繁 preempt，说明容量、调度参数、请求形态或缓存策略有问题。

## 51.2 preemption 解决的不是算力问题

LLM serving 里最容易耗尽的资源有两类：

1. GPU compute。
2. GPU memory，尤其是 KV cache blocks。

preemption 主要解决第二类。

假设 block pool 里一共有 1000 个 blocks。

当前 running 请求已经占用了 980 个。

下一轮 scheduler 想做两件事：

1. 给 running 请求继续 decode，可能每个请求都需要追加 KV。
2. 接纳 waiting 请求做 prefill，可能一次要分配很多 KV blocks。

如果 free blocks 不够，简单实现通常会失败：

```text
allocate blocks -> failed -> OOM / request failed / engine crash
```

更合理的实现是：

```text
allocate blocks -> failed -> preempt some running requests -> free their GPU KV -> retry scheduling
```

注意这里不是 GPU 算不过来。

而是 GPU KV 空间不够。

## 51.3 四个概念不要混

preemption 容易和 eviction、abort、backpressure 混在一起。

它们解决的问题不同。

| 概念 | 对象 | 结果 | 用户请求是否还继续 |
|---|---|---|---|
| preemption | running request | 暂停并释放或迁移 KV | 继续 |
| eviction | cached block | 淘汰无 active request 的缓存 | 不影响 active 请求 |
| abort | request | 主动终止请求并释放资源 | 不继续 |
| backpressure | waiting request / upstream | 限流、排队或拒绝新请求 | 可能继续，也可能拒绝 |

prefix cache 的 eviction 只应该动没有 active request 引用的 cached blocks。

preemption 动的是正在执行的 request。

abort 是直接结束请求。

backpressure 是在入口处减少进入 engine 的压力。

这四个机制可以同时存在，但不能互相替代。

## 51.4 preemption 的两种策略

被 preempt 的 request 要让出 GPU KV。

让出之后，后面怎么恢复？

常见有两种策略。

第一种是 recompute。

```text
preempt request -> 释放它的 GPU KV blocks -> request 回到 waiting
resume 时重新 prefill prompt + 已生成 tokens，重建 KV
```

第二种是 swap。

```text
preempt request -> 把 GPU KV blocks 拷贝到 CPU memory -> 释放 GPU KV blocks
resume 时把 CPU KV 拷回 GPU，继续 decode
```

对比：

| 策略 | 释放 GPU KV | 恢复成本 | 额外资源 | 适合场景 |
|---|---|---|---|---|
| recompute | 是 | 重新计算 prefill | GPU compute | prompt 不太长、CPU 带宽紧张、实现简单 |
| swap | 是 | CPU/GPU 拷贝 | CPU memory + PCIe/NVLink 带宽 | 上下文很长、重算很贵、带宽充足 |

教学版建议先实现 recompute。

原因是它的数据结构更简单，bug 更少，也更容易解释 scheduler 和 KV block 生命周期。

## 51.5 为什么通常优先讲 recompute

recompute 的最大优点是：不需要保存整段 KV。

被 preempt 后，只要保存 request 的逻辑状态：

1. prompt token ids。
2. 已生成 token ids。
3. sampling params。
4. stop criteria。
5. request id、arrival time、priority 等调度信息。

恢复时，把完整上下文拼出来：

```text
context_tokens = prompt_token_ids + output_token_ids
```

然后重新 prefill 到当前位置。

这会浪费计算，但避免了 CPU KV 存储、异步拷贝、pin memory、带宽竞争和 swap 状态一致性问题。

很多线上场景中，preemption 本来就应该是低频兜底机制。

既然低频，优先选择实现简单、稳定性高的 recompute 是合理的。

## 51.6 swap 为什么更复杂

swap 看起来更节省计算。

但工程复杂度明显更高。

你需要考虑：

1. CPU KV buffer 如何分配。
2. GPU block 到 CPU block 的映射如何维护。
3. swap out 和 swap in 是否异步。
4. 拷贝时 request 是否还能被 scheduler 选中。
5. CPU 内存不够怎么办。
6. PCIe/NVLink 带宽被 KV 拷贝打满怎么办。
7. swap in 失败后 request 状态如何回滚。
8. block table 里的物理 block id 何时更新。
9. prefix cache block 是否允许 swap。
10. 多 GPU / TP 场景下每个 rank 的 KV 是否一致迁移。

所以 swap 不是“把 tensor `.cpu()` 一下”这么简单。

它是一套内存层级管理机制。

教学版可以先把 swap 作为接口和状态机讲清楚，不急着实现高性能异步拷贝。

## 51.7 request 状态机

第 48 章的 continuous batching 已经有 waiting 和 running。

加入 preemption 后，建议显式增加 preempted 状态。

```text
WAITING -> RUNNING -> FINISHED
               |
               v
          PREEMPTED
               |
               v
            WAITING
```

如果实现 swap，可以再细分：

```text
RUNNING -> SWAPPING_OUT -> SWAPPED -> SWAPPING_IN -> RUNNING
```

教学版 recompute 可以简单一点：

```python
class RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"
```

request 里需要增加几个字段：

```python
class RequestState:
    def __init__(self, request_id, prompt_token_ids):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = []
        self.block_table = []
        self.num_computed_tokens = 0
        self.status = RequestStatus.WAITING
        self.preempted_count = 0
        self.last_preempted_at = None
```

关键点是：

```text
preemption 释放的是 GPU KV，不应该丢失 request 的逻辑 token 序列。
```

## 51.8 recompute preemption 的最小实现

recompute 版本的 preempt 操作很短。

伪代码：

```python
def preempt_by_recompute(request, block_manager):
    block_manager.free_request_blocks(request)

    request.block_table = []
    request.num_computed_tokens = 0
    request.status = RequestStatus.PREEMPTED
    request.preempted_count += 1
```

如果这个 request 命中过 prefix cache，释放时要注意 ref count。

```python
def free_request_blocks(request):
    for block_id in request.block_table:
        block = self.blocks[block_id]
        block.active_refs -= 1

        if block.active_refs == 0 and not block.cached:
            self.free_blocks.append(block_id)

    request.block_table = []
```

cached block 不应该因为 request 被 preempt 就直接回到 free list。

它只释放 active ref。

如果还有 cache ref，它仍然留在 prefix cache 中。

## 51.9 recompute 恢复流程

被 preempt 的 request 不应该回到队尾当成全新请求，也不应该立刻无条件插队。

一种简单做法是把它放回 waiting queue，但保留优先级信息。

恢复时：

```python
def resume_recompute(request):
    request.status = RequestStatus.WAITING
    request.block_table = []
    request.num_computed_tokens = 0
```

scheduler 下次选中它时，按照普通 prefill 处理。

但 prefill 的输入不只是 prompt。

它应该是：

```python
context_tokens = request.prompt_token_ids + request.output_token_ids
```

也就是说，已经生成过的 tokens 也要参与 recompute。

否则恢复后的 KV 只覆盖 prompt，不包含已生成输出的历史上下文。

这是 recompute 最常见的 bug。

## 51.10 为什么 output tokens 也要重算

假设一个请求：

```text
prompt: P0 P1 P2
already generated: Y0 Y1 Y2
```

preempt 发生在准备生成 Y3 之前。

此时 KV cache 里保存的是：

```text
P0 P1 P2 Y0 Y1 Y2
```

如果恢复时只 prefill prompt：

```text
P0 P1 P2
```

然后直接 decode Y3，模型看不到 Y0/Y1/Y2 的 KV。

结果会错。

所以 recompute 的上下文必须包含 prompt 和已经生成的输出。

```text
recompute context = prompt + generated outputs
```

这也是为什么 preemption 不是免费的。

请求已经生成得越长，recompute 成本越高。

## 51.11 scheduler 什么时候触发 preemption

最小实现里有两个触发点。

第一个触发点：接纳 waiting 请求前，发现 KV blocks 不够。

```python
if not block_manager.can_allocate(num_required_blocks):
    victim = select_victim(running_requests)
    preempt_by_recompute(victim, block_manager)
```

第二个触发点：running 请求 decode 追加 block 时，发现没有 free block。

```python
if request_needs_new_block(req) and not block_manager.has_free_block():
    victim = select_victim(running_requests)
    preempt_by_recompute(victim, block_manager)
```

更实际的 scheduler 会在每轮调度前做预算：

```text
required_blocks = blocks needed by selected prefill chunks + blocks needed by decode growth
```

如果预算不够，再决定是否：

1. 减少本轮 prefill chunk。
2. 暂缓接纳 waiting 请求。
3. 淘汰 prefix cache blocks。
4. preempt running requests。
5. 触发 backpressure。

preemption 应该排在比较靠后的位置。

因为它会伤害延迟。

## 51.12 victim 选择策略

victim 就是被 preempt 的请求。

最简单策略是选最后进入 running 的请求。

```python
def select_victim(running_requests):
    return running_requests[-1]
```

但真实系统里要更谨慎。

常见策略包括：

1. 优先 preempt 已经 preempt 次数少的请求。
2. 优先 preempt 低优先级请求。
3. 优先 preempt recompute 成本低的请求。
4. 避免 preempt 已经接近完成的请求。
5. 避免反复 preempt 同一个请求。
6. 避免 preempt latency-sensitive 请求。

一个教学版评分函数可以这样写：

```python
def victim_score(req):
    context_len = len(req.prompt_token_ids) + len(req.output_token_ids)
    return (
        req.priority * 1000000
        + req.preempted_count * 10000
        + context_len
    )

def select_victim(running_requests):
    return min(running_requests, key=victim_score)
```

这里分数越小越适合被 preempt。

低优先级、preempt 次数少、上下文短的请求更容易被选中。

实际实现中不要迷信某个公式，要根据业务 SLO 调整。

## 51.13 preempt 多少个请求

不要一遇到空间不足就只 preempt 一个请求然后盲目重试。

更好的做法是计算缺口。

```python
def preempt_until_enough(required_blocks, running_requests, block_manager):
    victims = []

    while block_manager.num_free_blocks() < required_blocks:
        victim = select_victim(running_requests)
        if victim is None:
            break

        running_requests.remove(victim)
        preempt_by_recompute(victim, block_manager)
        victims.append(victim)

    return victims
```

但这里有一个细节：

```text
preempt victim 释放的 blocks 未必等于它 block_table 的长度。
```

如果 victim 的某些 blocks 是 cached blocks，释放 active ref 后它们可能仍然被 prefix cache 持有。

这些 blocks 不会回到 free list。

所以判断要以实际 free block 数为准，而不是估算。

## 51.14 preemption 和 prefix cache 的关系

prefix cache 会影响 preemption。

第一，它占用 KV blocks。

cached blocks 没有 active request 使用时，仍然可能留在 block pool 中。

这会降低 free blocks，增加 preemption 概率。

第二，它也能降低 recompute 成本。

如果被 preempt 的 request 恢复时，prompt prefix 命中 prefix cache，就不需要从 0 重算全部上下文。

恢复路径可以变成：

```text
find cached prefix -> 引用 cached blocks -> recompute suffix context
```

第三，资源紧张时通常应该先考虑 cache eviction，再考虑 preempt active request。

推荐顺序是：

1. 降低本轮接纳的新 prefill tokens。
2. 淘汰没有 active request 引用的 cached blocks。
3. 仍然不够时，再 preempt running requests。

因为 eviction 影响未来命中率，preemption 直接影响当前请求延迟。

两者都不是免费操作。

## 51.15 preemption 和 chunked prefill 的关系

chunked prefill 可以降低 preemption 频率。

如果一个 32K prompt 一次性 prefill，需要一次性分配大量 KV blocks。

在高并发下，这很容易挤爆 block pool。

chunked prefill 把它拆成多轮：

```text
32K prompt -> 2K + 2K + ... + 2K
```

scheduler 每轮只需要为当前 chunk 做预算。

这让系统有机会在 iteration boundary 上重新观察 free blocks、finished requests 和 waiting queue。

但 chunked prefill 不是万能的。

完整 prompt 的 KV 最终还是要占空间。

它只是降低单轮峰值压力，不会改变总 KV 需求。

## 51.16 swap 的最小数据结构

如果要实现 swap，需要在 request 上记录 CPU KV 位置。

教学版可以抽象成：

```python
class RequestState:
    def __init__(self, request_id, prompt_token_ids):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = []
        self.block_table = []
        self.cpu_block_table = []
        self.status = RequestStatus.WAITING
```

再增加一个 CPU block manager：

```python
class CPUBlockManager:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))

    def allocate(self, n):
        if len(self.free_blocks) < n:
            return None
        blocks = self.free_blocks[:n]
        self.free_blocks = self.free_blocks[n:]
        return blocks
```

GPU block table 和 CPU block table 都是逻辑 block index 到物理 block id 的映射。

区别只是物理位置不同。

## 51.17 swap out 流程

swap out 的伪代码：

```python
def swap_out(request, gpu_mgr, cpu_mgr, kv_copy):
    cpu_blocks = cpu_mgr.allocate(len(request.block_table))
    if cpu_blocks is None:
        return False

    for gpu_block, cpu_block in zip(request.block_table, cpu_blocks):
        kv_copy.gpu_to_cpu(gpu_block, cpu_block)

    gpu_mgr.free_request_blocks(request)
    request.cpu_block_table = cpu_blocks
    request.block_table = []
    request.status = RequestStatus.SWAPPED
    return True
```

真实系统通常不会同步逐个拷贝。

它会考虑异步 copy、stream、pinned memory 和批量拷贝。

但教学版先把状态变化讲清楚即可。

## 51.18 swap in 流程

swap in 的伪代码：

```python
def swap_in(request, gpu_mgr, cpu_mgr, kv_copy):
    gpu_blocks = gpu_mgr.allocate(len(request.cpu_block_table))
    if gpu_blocks is None:
        return False

    for cpu_block, gpu_block in zip(request.cpu_block_table, gpu_blocks):
        kv_copy.cpu_to_gpu(cpu_block, gpu_block)

    cpu_mgr.free(request.cpu_block_table)
    request.cpu_block_table = []
    request.block_table = gpu_blocks
    request.status = RequestStatus.RUNNING
    return True
```

swap in 成功后，request 可以继续 decode。

如果 swap in 失败，不应该丢掉 CPU KV。

request 仍然保持 SWAPPED 状态，等待下一轮资源可用。

## 51.19 swap 的一致性问题

swap 必须保证逻辑 token 位置和 KV block 顺序一致。

不能因为 GPU block id 变化就改变 block table 的逻辑顺序。

例如 swap out 前：

```text
logical block 0 -> GPU block 7
logical block 1 -> GPU block 3
logical block 2 -> GPU block 12
```

swap in 后可以变成：

```text
logical block 0 -> GPU block 20
logical block 1 -> GPU block 5
logical block 2 -> GPU block 9
```

物理 id 可以变。

逻辑顺序不能变。

BatchBuilder 后续生成 attention metadata 时，必须用新的 `request.block_table`。

## 51.20 recompute 和 swap 如何选择

一个实用判断：

```text
如果 preemption 低频，优先 recompute。
如果上下文极长且重算成本高，可以考虑 swap。
```

更具体一点：

适合 recompute：

1. 中短上下文聊天。
2. preemption 只是偶发兜底。
3. CPU 内存或 PCIe 带宽紧张。
4. 系统更看重实现稳定性。
5. prefix cache 命中率较高，重算 suffix 不大。

适合 swap：

1. 长上下文请求多。
2. preempt 后重算成本非常高。
3. CPU 内存充足。
4. GPU/CPU 传输带宽充足。
5. engine 已经有成熟的多级 KV cache 管理。

不要把 swap 当成无成本优化。

swap 可能把瓶颈从 GPU memory 转移到 CPU memory 和总线带宽。

## 51.21 对延迟指标的影响

preemption 会影响多个指标。

对被 preempt 的请求：

1. E2E latency 变差。
2. 如果已经开始 streaming，后续 token 间隔可能出现大 gap。
3. recompute 会增加恢复后的 TTFT-like delay。
4. swap 会增加 copy delay。

对其他请求：

1. 可能避免 OOM。
2. 可能释放 KV，让高优先级请求继续。
3. 也可能因为 recompute 或 swap 占资源，造成整体抖动。

所以 preemption 指标必须和 TTFT、TPOT、E2E latency 一起看。

只看吞吐可能会误判。

## 51.22 需要记录哪些日志

每次 preemption 都应该打结构化日志。

建议字段：

| 字段 | 含义 |
|---|---|
| request_id | 被 preempt 的请求 |
| reason | 触发原因，例如 no_free_blocks |
| strategy | recompute 或 swap |
| num_blocks_freed | 实际释放 GPU blocks 数 |
| num_context_tokens | prompt + output token 数 |
| preempted_count | 该请求累计 preempt 次数 |
| free_blocks_before | preempt 前 free blocks |
| free_blocks_after | preempt 后 free blocks |
| running_queue_size | running 请求数 |
| waiting_queue_size | waiting 请求数 |

日志示例：

```text
preempt request=req-42 strategy=recompute reason=no_free_blocks \
context_tokens=4096 blocks_freed=256 free_blocks_before=3 free_blocks_after=259 preempted_count=1
```

这类日志对排查线上延迟尖刺非常有用。

## 51.23 需要暴露哪些指标

建议指标：

| 指标 | 含义 |
|---|---|
| preemption_total | preemption 总次数 |
| preemption_recompute_total | recompute 次数 |
| preemption_swap_total | swap 次数 |
| preempted_requests_total | 被 preempt 的请求数 |
| preemption_repeated_requests | 被多次 preempt 的请求数 |
| preemption_freed_blocks | preemption 实际释放 blocks 数 |
| recompute_tokens_total | recompute 重算 tokens 数 |
| swap_in_bytes_total | swap in 字节数 |
| swap_out_bytes_total | swap out 字节数 |
| swap_in_latency_ms | swap in 延迟 |
| swap_out_latency_ms | swap out 延迟 |
| free_kv_blocks | 当前 free blocks |
| preemption_rate | 单位时间 preemption 次数 |

最关键的是：

```text
preemption_rate 不应该长期处于高位。
```

如果它持续升高，说明系统在靠牺牲尾延迟维持运行。

## 51.24 压测场景一：突发长 prompt

workload：

```text
短请求：prompt 128 tokens，output 128 tokens，持续到达
突发请求：每隔一段时间注入 prompt 16K tokens，output 256 tokens
```

观察：

1. free blocks 是否快速下降。
2. 长 prompt 是否触发 preemption。
3. 短请求 TPOT 是否出现尖刺。
4. preempted request 的 E2E latency 是否明显变差。
5. chunked prefill 后 preemption 是否减少。

这个场景用来验证 scheduler 在长 prompt 冲击下是否能保持服务不崩。

## 51.25 压测场景二：长输出 decode 压力

workload：

```text
prompt 512 tokens
output 4K tokens
高并发持续到达
```

这个场景里 KV cache 压力不是一次性来自 prefill，而是 decode 逐步增长。

观察：

1. running requests 的 block_table 是否持续变长。
2. decode 追加新 block 时是否触发 preemption。
3. victim 是否总是同一批请求。
4. repeated preemption 是否过高。
5. max_num_seqs 降低后 preemption 是否缓解。

它可以暴露 victim 选择策略是否公平。

## 51.26 压测场景三：prefix cache 占满 blocks

workload：

```text
大量共享 system prompt 请求，让 prefix cache 留住很多 blocks
随后注入随机长 prompt 请求
```

观察：

1. cached blocks 是否占据大量 block pool。
2. free blocks 不足时是否先执行 cache eviction。
3. eviction 后是否减少 preemption。
4. cached block ref count 是否正确。
5. cache map 是否出现悬挂 block id。

这个场景用来验证 prefix cache eviction 和 preemption 的优先级关系。

## 51.27 常见 bug

bug 一：preempt 后丢失 output tokens。

```text
结果：recompute 恢复时上下文不完整，后续生成错误。
```

bug 二：preempt 后只重算 prompt，不重算已生成 tokens。

```text
结果：模型继续 decode 时看不到历史输出。
```

bug 三：释放 cached blocks 时直接放回 free list。

```text
结果：prefix cache map 指向的 block 被新请求覆盖。
```

bug 四：victim 选择总是同一个请求。

```text
结果：某些请求被反复 preempt，E2E latency 极差甚至饥饿。
```

bug 五：swap in 失败后清空 CPU block table。

```text
结果：request 无法恢复，KV 丢失。
```

bug 六：swap in 后没有更新 GPU block table。

```text
结果：attention kernel 读旧 block id 或空 block。
```

bug 七：preemption 计数没有暴露指标。

```text
结果：线上只看到 latency 抖动，不知道根因是 KV 不足。
```

bug 八：把 preemption 当成常规调度优化。

```text
结果：系统长期高频重算或 swap，尾延迟越来越差。
```

## 51.28 面试高频问题

问题一：vLLM 里的 preemption 是什么？

回答要点：当 KV cache 空间不足以继续服务当前调度集合时，engine 可以 preempt 一部分 running 请求，释放它们占用的 GPU KV，后续再通过 recompute 或 swap 恢复。它是避免 OOM 的鲁棒性机制，不是性能优化。

问题二：recompute 和 swap 有什么区别？

回答要点：recompute 会释放 GPU KV，恢复时根据 prompt 和已生成 tokens 重新 prefill 重建 KV；swap 会把 GPU KV 拷到 CPU，恢复时再拷回 GPU。recompute 消耗计算但实现简单，swap 节省重算但消耗 CPU 内存和传输带宽，状态管理更复杂。

问题三：为什么 recompute 恢复时要包含 output tokens？

回答要点：因为被 preempt 时 KV cache 表示的是 prompt 加已生成输出的完整上下文。如果恢复时只重算 prompt，后续 decode 看不到已经生成 tokens 的历史 KV，生成语义会错。

问题四：频繁 preemption 说明什么？

回答要点：通常说明 KV cache budget 不够、并发过高、请求上下文过长、`max_num_seqs` 或 `max_num_batched_tokens` 太激进、prefix cache 占用过多，或者缺少合理的 chunked prefill 和 admission control。

问题五：preemption 和 prefix cache eviction 谁优先？

回答要点：一般应先减少本轮调度压力，再淘汰没有 active request 引用的 cached blocks，最后才 preempt running requests。eviction 影响未来复用，preemption 直接影响当前请求延迟。

## 51.29 标准回答模板

如果面试官问“KV cache 不够时 serving engine 怎么办”，可以这样回答：

```text
我会先把 KV cache 作为 scheduler 的核心资源预算。每轮调度前，scheduler 根据 selected prefill chunks、decode 增长和 max running requests 估算需要的 KV blocks。如果 free blocks 不够，先降低本轮 prefill token budget 或推迟接纳新请求；如果启用了 prefix cache，再优先淘汰没有 active request 引用的 cached blocks。

如果仍然不够，就需要 preemption。preemption 是把部分 running request 暂停，释放它们占用的 GPU KV，后续再恢复。恢复策略主要有 recompute 和 swap。recompute 会丢弃 GPU KV，保留 request 的 prompt tokens、已生成 output tokens、sampling params 和状态，恢复时重新 prefill prompt + output tokens 重建 KV；swap 会把 GPU KV 拷到 CPU，恢复时再拷回 GPU，节省重算但增加 CPU 内存、带宽和状态一致性复杂度。

victim 选择不能随便做。一般要考虑请求优先级、上下文长度、已 preempt 次数、是否接近完成、是否 latency-sensitive，避免反复 preempt 同一个请求。preemption 后要记录结构化日志和指标，比如 preemption_total、recompute_tokens_total、swap bytes、free blocks before/after、repeated preemption 和被 preempt 请求的 E2E latency。

最后要强调，preemption 是防 OOM 的鲁棒性机制，不是性能优化。如果线上频繁 preemption，应该回头调 max_num_seqs、max_num_batched_tokens、max_model_len、gpu_memory_utilization、chunked prefill、prefix cache eviction 和入口限流，而不是依赖 preemption 长期维持系统。
```

## 51.30 Preemption / Recompute / Swap 公式、状态机和可运行 demo

preemption 的触发条件可以先抽象成 KV block 缺口：

```math
F_t < B_{\mathrm{need},t}
```

其中 `F_t` 是当前 free GPU KV blocks，`B_{\mathrm{need},t}` 是本轮想调度的 prefill / decode 额外 block 数。

对 recompute 策略，被 preempt 请求 `i` 的恢复重算 token 数是：

```math
C_{\mathrm{recompute},i}=|X_i|+|Y_i|
```

其中 `X_i` 是 prompt tokens，`Y_i` 是已经生成的 output tokens。

对 swap 策略，如果被迁移的 block 数是 `B_{\mathrm{swap}}`，单 block page bytes 是 `P_{\mathrm{page}}`，swap 数据量是：

```math
M_{\mathrm{swap}}=B_{\mathrm{swap}}P_{\mathrm{page}}
```

单位窗口内 preemption rate：

```math
R_{\mathrm{preempt}}=\frac{N_{\mathrm{preempt}}}{\max(1,\Delta t)}
```

最终门禁：

```math
G_{\mathrm{preempt}}=G_{\mathrm{pressure}}G_{\mathrm{evict}}G_{\mathrm{victim}}G_{\mathrm{recompute}}G_{\mathrm{swap}}G_{\mathrm{metric}}
```

下面这个 0 依赖 demo 模拟一个最小调度回合：

1. GPU block pool 剩余空间不足，waiting 请求需要 4 个 blocks。
2. scheduler 先淘汰 cached-free block，仍然不够。
3. scheduler 选择低优先级 running request 做 recompute preemption，释放 GPU KV。
4. 被 preempt 请求保留 prompt + output tokens，后续恢复时重建完整上下文。
5. 另一个请求演示 swap out / swap in；swap in 失败时 CPU block table 不丢失。

```python
from dataclasses import dataclass, field


@dataclass
class RequestState:
    request_id: str
    prompt_tokens: list
    output_tokens: list
    priority: int
    block_table: list = field(default_factory=list)
    cpu_block_table: list = field(default_factory=list)
    status: str = "WAITING"
    num_computed_tokens: int = 0
    preempted_count: int = 0
    last_preempted_at: int = None

    def context_tokens(self):
        return self.prompt_tokens + self.output_tokens


class GPUBlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.cached_free_blocks = []

    def required_blocks(self, token_count):
        return (token_count + self.block_size - 1) // self.block_size

    def allocate_for_tokens(self, req, token_count):
        needed = self.required_blocks(token_count)
        if len(self.free_blocks) < needed:
            return False
        req.block_table = [self.free_blocks.pop(0) for _ in range(needed)]
        req.num_computed_tokens = token_count
        req.status = "RUNNING"
        return True

    def hold_cached_free_block(self):
        block_id = self.free_blocks.pop(0)
        self.cached_free_blocks.append(block_id)
        return block_id

    def evict_cached_free(self):
        if not self.cached_free_blocks:
            return []
        block_id = self.cached_free_blocks.pop(0)
        self.free_blocks.append(block_id)
        return [block_id]

    def free_request(self, req):
        released = list(req.block_table)
        self.free_blocks.extend(released)
        req.block_table = []
        req.num_computed_tokens = 0
        return released


class CPUBlockManager:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))

    def allocate(self, count):
        if len(self.free_blocks) < count:
            return None
        blocks = self.free_blocks[:count]
        self.free_blocks = self.free_blocks[count:]
        return blocks

    def free(self, blocks):
        self.free_blocks.extend(blocks)


def select_victim(running):
    def score(req):
        return (req.priority, req.preempted_count, len(req.context_tokens()))

    return min(running, key=score)


def preempt_by_recompute(req, gpu_mgr, step):
    released = gpu_mgr.free_request(req)
    req.status = "PREEMPTED"
    req.preempted_count += 1
    req.last_preempted_at = step
    return released, len(req.context_tokens())


def resume_by_recompute(req, gpu_mgr):
    context = req.context_tokens()
    ok = gpu_mgr.allocate_for_tokens(req, token_count=len(context))
    return ok, context


def swap_out(req, gpu_mgr, cpu_mgr):
    cpu_blocks = cpu_mgr.allocate(len(req.block_table))
    if cpu_blocks is None:
        return False, []
    copy_plan = list(zip(req.block_table, cpu_blocks))
    gpu_mgr.free_request(req)
    req.cpu_block_table = cpu_blocks
    req.status = "SWAPPED"
    return True, copy_plan


def swap_in(req, gpu_mgr, cpu_mgr):
    if len(gpu_mgr.free_blocks) < len(req.cpu_block_table):
        return False, []
    gpu_blocks = [gpu_mgr.free_blocks.pop(0) for _ in req.cpu_block_table]
    copy_plan = list(zip(req.cpu_block_table, gpu_blocks))
    cpu_mgr.free(req.cpu_block_table)
    req.cpu_block_table = []
    req.block_table = gpu_blocks
    req.status = "RUNNING"
    return True, copy_plan


gpu = GPUBlockManager(num_blocks=8, block_size=2)
low = RequestState("low", [101, 102, 103], [201, 202], priority=0)
high = RequestState("high", [301, 302], [401], priority=10)
burst = RequestState("burst", list(range(700, 708)), [], priority=5)

gpu.allocate_for_tokens(low, len(low.context_tokens()))
gpu.allocate_for_tokens(high, len(high.context_tokens()))
cached_block = gpu.hold_cached_free_block()
free_before = len(gpu.free_blocks)
required_blocks = gpu.required_blocks(len(burst.prompt_tokens))

evicted_cache_blocks = []
victims = []
freed_by_preemption = 0
recompute_tokens_total = 0
if free_before < required_blocks:
    evicted_cache_blocks = gpu.evict_cached_free()
if len(gpu.free_blocks) < required_blocks:
    victim = select_victim([low, high])
    released, recompute_tokens = preempt_by_recompute(victim, gpu, step=7)
    victims.append(victim.request_id)
    freed_by_preemption += len(released)
    recompute_tokens_total += recompute_tokens

gpu.allocate_for_tokens(burst, len(burst.prompt_tokens))
burst_blocks = list(burst.block_table)
gpu.free_request(burst)
resume_ok, low_recompute_context = resume_by_recompute(low, gpu)

swap_gpu = GPUBlockManager(num_blocks=3, block_size=2)
swap_cpu = CPUBlockManager(num_blocks=3)
swap_req = RequestState("swap", [501, 502, 503], [601], priority=1)
swap_gpu.allocate_for_tokens(swap_req, len(swap_req.context_tokens()))
swap_out_ok, swap_out_plan = swap_out(swap_req, swap_gpu, swap_cpu)
failed_gpu = GPUBlockManager(num_blocks=0, block_size=2)
swap_in_fail_ok, _ = swap_in(swap_req, failed_gpu, swap_cpu)
cpu_table_after_failed_swap_in = list(swap_req.cpu_block_table)
swap_in_ok, swap_in_plan = swap_in(swap_req, swap_gpu, swap_cpu)

metrics = {
    "preemption_total": len(victims),
    "preemption_recompute_total": len(victims),
    "preemption_swap_total": int(swap_out_ok and swap_in_ok),
    "preemption_freed_blocks": freed_by_preemption,
    "recompute_tokens_total": recompute_tokens_total,
    "swap_out_blocks": len(swap_out_plan),
    "swap_in_blocks": len(swap_in_plan),
    "cache_evictions_before_preempt": len(evicted_cache_blocks),
}
summary = {
    "free_before": free_before,
    "required_blocks": required_blocks,
    "cached_block": cached_block,
    "evicted_cache_blocks": evicted_cache_blocks,
    "free_after_eviction": free_before + len(evicted_cache_blocks),
    "victims": victims,
    "freed_by_preemption": freed_by_preemption,
    "burst_blocks": burst_blocks,
    "low_recompute_context": low_recompute_context,
    "low_resume_ok": resume_ok,
    "low_resume_blocks": list(low.block_table),
    "low_preempted_count": low.preempted_count,
    "swap_out_plan": swap_out_plan,
    "swap_in_fail_ok": swap_in_fail_ok,
    "cpu_table_after_failed_swap_in": cpu_table_after_failed_swap_in,
    "swap_in_plan": swap_in_plan,
    "swap_status_after_in": swap_req.status,
    "metrics": metrics,
}
gates = {
    "kv_pressure_detected": free_before < required_blocks,
    "eviction_before_preemption": evicted_cache_blocks == [5] and victims == ["low"],
    "victim_policy_ready": victims == ["low"],
    "recompute_context_preserved": low_recompute_context == [101, 102, 103, 201, 202],
    "recompute_resume_ready": resume_ok and len(low.block_table) == 3 and low.status == "RUNNING",
    "swap_failure_preserves_cpu": swap_in_fail_ok is False and cpu_table_after_failed_swap_in == [0, 1],
    "swap_in_ready": swap_in_ok and swap_req.status == "RUNNING" and swap_req.cpu_block_table == [],
    "metrics_ready": metrics["preemption_total"] == 1 and metrics["recompute_tokens_total"] == 5,
}
gates["preemption_upgrade_gate"] = all(gates.values())

print("preemption_upgrade_summary=", summary)
print("preemption_upgrade_gates=", gates)
```

一次运行的核心输出类似：

```text
preemption_upgrade_summary= {'free_before': 2, 'required_blocks': 4, 'cached_block': 5, 'evicted_cache_blocks': [5], 'free_after_eviction': 3, 'victims': ['low'], 'freed_by_preemption': 3, 'burst_blocks': [6, 7, 5, 0], 'low_recompute_context': [101, 102, 103, 201, 202], 'low_resume_ok': True, 'low_resume_blocks': [1, 2, 6], 'low_preempted_count': 1, 'swap_out_plan': [(0, 0), (1, 1)], 'swap_in_fail_ok': False, 'cpu_table_after_failed_swap_in': [0, 1], 'swap_in_plan': [(0, 2), (1, 0)], 'swap_status_after_in': 'RUNNING', 'metrics': {'preemption_total': 1, 'preemption_recompute_total': 1, 'preemption_swap_total': 1, 'preemption_freed_blocks': 3, 'recompute_tokens_total': 5, 'swap_out_blocks': 2, 'swap_in_blocks': 2, 'cache_evictions_before_preempt': 1}}
preemption_upgrade_gates= {'kv_pressure_detected': True, 'eviction_before_preemption': True, 'victim_policy_ready': True, 'recompute_context_preserved': True, 'recompute_resume_ready': True, 'swap_failure_preserves_cpu': True, 'swap_in_ready': True, 'metrics_ready': True, 'preemption_upgrade_gate': True}
```

这个 demo 证明了几个关键点：

1. KV 不足时先 eviction，再 preempt active request。
2. victim 选择没有抢占高优先级请求。
3. recompute preemption 释放 GPU KV，但保留 prompt + output tokens。
4. 恢复时重算上下文是 `[prompt, output]`，不是只重算 prompt。
5. swap in 失败后 CPU block table 仍然保留，request 不会丢状态。
6. swap in 成功后 GPU block id 可以变化，但逻辑 block 顺序仍由新的 block table 表示。
7. 指标里能看到 preemption 次数、释放 blocks、重算 tokens 和 swap blocks。

## 51.31 小练习

1. 给 `RequestState` 增加 `status`、`preempted_count`、`last_preempted_at` 字段。
2. 实现 `preempt_by_recompute(request, block_manager)`。
3. 确保 preempt 后只释放 GPU KV，不丢失 prompt tokens 和 output tokens。
4. 恢复时用 `prompt_token_ids + output_token_ids` 作为 recompute context。
5. 修改 scheduler，在 free blocks 不足时触发 preemption。
6. 实现一个最简单的 `select_victim`。
7. 给 victim 选择增加 `preempted_count` 约束，避免反复 preempt 同一请求。
8. 增加 `preemption_total` 和 `recompute_tokens_total` 指标。
9. 增加 preemption 结构化日志。
10. 构造长 prompt 突发压测，观察 preemption 是否发生。
11. 开启 chunked prefill，对比 preemption 次数变化。
12. 构造 prefix cache 占用 block pool 的场景，验证先 eviction 再 preemption。
13. 设计 `CPUBlockManager`，实现同步版 swap out。
14. 实现同步版 swap in，并确保失败时不清空 CPU block table。
15. 写一段面试回答：为什么 preemption 不是性能优化？

## 51.32 本章总结

preemption 是 serving engine 在 KV cache 压力下保护系统稳定性的机制。

它的核心不是让系统更快，而是在资源不足时避免直接 OOM，并让请求后续仍有机会继续执行。

最小教学版建议先实现 recompute：preempt 时释放 request 的 GPU KV blocks，保留 prompt tokens、output tokens 和采样状态；恢复时重新 prefill `prompt + output` 重建 KV。

swap 可以减少重算，但需要 CPU KV buffer、GPU/CPU block 映射、异步拷贝、带宽控制和失败回滚，工程复杂度更高。

preemption 必须和 scheduler、paged KV cache、prefix cache eviction、chunked prefill、admission control 和指标体系一起设计。

频繁 preemption 是危险信号。

真正的优化方向通常是降低并发压力、缩短上下文、调整 `max_num_seqs` 和 `max_num_batched_tokens`、改善 chunked prefill、控制 prefix cache 占用，或者扩充 KV cache budget。

下一章可以继续升级 mini engine：把这些机制串成一个完整的调度循环，并讨论如何用系统化 benchmark 验证每次改造真的有效。
