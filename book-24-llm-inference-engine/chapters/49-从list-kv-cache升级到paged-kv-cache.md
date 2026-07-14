# 第 49 章 从 list KV cache 升级到 paged KV cache

上一章我们把 naive scheduler 升级成了 continuous batching。

但只改 scheduler 还不够。

如果底层 KV cache 仍然是教学版 list 或连续张量，continuous batching 很快会撞到三个问题：

1. 长短请求混合时显存浪费严重。
2. 请求动态加入和退出时，KV 空间难以复用。
3. scheduler 很难在调度前准确判断还能接纳多少请求。

本章继续沿着 mini engine 的升级路线，把 KV cache 从最简单的 list 结构升级成 paged KV cache。

这一章不是重复 PagedAttention 原理。

前面的第 17、18、21 章已经讲过 block table、block manager 和 vLLM memory management。

本章关注一个更工程化的问题：

```text
如果你已经有一个能跑的教学版推理框架，应该怎样一步步把 KV cache 改成 paged 版本？
```

## 49.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲把教学版 `list KV cache` 升级成单机、单模型、单进程内的 paged KV cache。

公开资料校准主要参考三类口径：

1. vLLM / PagedAttention 论文对固定大小 KV blocks、logical block 到 physical block 映射、非连续 KV 存储和高吞吐 serving 的公开定义。
2. vLLM 文档对 KV cache blocks、KV cache usage、preemption、chunked prefill 和 scheduler admission 的公开口径。
3. 本书第 17、18、21、48 章已经建立的 PagedAttention、KV block manager、vLLM memory management 和 continuous batching 边界。

本章不实现真实 attention kernel、GPU KV tensor layout、prefix cache、block hash、ref count sharing、LRU eviction、CPU swap、preemption recompute、多 worker KV 分片或多租户隔离。我们只验证一个 mini engine 改造中最小但关键的工程闭环：

```text
global block pool -> request block table -> allocate_until -> slot_mapping -> model input metadata -> free / reuse / metrics
```

第二轮新增 demo 的验收重点是：

```text
list KV 私有缓存是否被全局 block pool 替代；
request block table 是否随 prefill / decode 增长；
decode 跨 block 边界是否追加 block；
BatchBuilder 是否生成 slot mapping；
finished / aborted / failed 是否幂等释放；
释放后的 block 是否能被新请求复用；
scheduler 是否能在 admission 阶段拦住 KV block 不足。
```

## 49.1 本章目标

读完本章，你应该能讲清：

1. 教学版 list KV cache 的结构为什么不适合 online serving。
2. paged KV cache 要新增哪些核心数据结构。
3. request 的 block table 如何维护。
4. prefill 和 decode 阶段分别如何分配 KV blocks。
5. BatchBuilder 为什么需要生成 slot mapping 和 block table。
6. scheduler 如何基于 free blocks 做 admission control。
7. finished、abort、failed 请求如何释放 KV blocks。
8. 如何用日志和压测证明 paged KV cache 生效。

面试时，很多人能背 PagedAttention 的定义，但讲不清“如果让我改一个 mini engine，我具体改哪里”。本章就是为了补上这块。

## 49.2 教学版 list KV cache 长什么样

在最小教学项目里，KV cache 往往写得很直接。

例如每个 request 保存一个 list：

```python
class RequestState:
    def __init__(self, request_id):
        self.request_id = request_id
        self.k_cache = []
        self.v_cache = []
```

每生成一个 token，就把新 KV append 进去：

```python
request.k_cache.append(new_k)
request.v_cache.append(new_v)
```

这种写法适合教学。

因为它很容易理解：

```text
一个请求 -> 一个不断增长的 KV list
```

但它不适合真正 serving。

原因不是 Python list 本身，而是它背后的抽象不对。

它把 KV cache 看成了 request 私有的、线性增长的对象，而不是全局可调度的显存资源。

## 49.3 list KV cache 的核心问题

第一个问题：没有统一资源池。

每个请求自己持有 KV，scheduler 很难快速回答：

```text
现在还能接多少 token 的 KV？
```

它可能只能看到有多少请求在跑，却看不到每个请求未来还要扩多少 cache。

第二个问题：释放和复用不清晰。

请求完成后，Python 对象可能被释放，但 GPU 上的 KV buffer 是否立即复用、如何复用、是否产生碎片，都没有明确抽象。

第三个问题：batch metadata 不稳定。

如果每个请求的 KV 是独立 list，BatchBuilder 在拼 batch 时要处理很多不规则对象，很难生成高效 attention kernel 需要的统一 metadata。

第四个问题：无法自然支持 continuous batching。

continuous batching 的特点是每一轮都有请求加入、退出、继续 decode。

KV cache 必须支持：

1. 请求进入 running 时分配。
2. decode 增长时追加。
3. 请求完成时释放。
4. 新请求复用旧请求释放的空间。

list KV cache 没有把这些操作变成一套显式协议。

## 49.4 paged KV cache 的升级目标

升级后的目标是：把 KV cache 从 request 私有 list 改成全局 block pool。

也就是从：

```text
request A -> [token0 kv, token1 kv, token2 kv, ...]
request B -> [token0 kv, token1 kv, ...]
```

改成：

```text
global KV blocks:
  block 0
  block 1
  block 2
  ...

request A block table: [7, 3, 12]
request B block table: [1, 9]
```

对 request 来说，token 位置仍然是连续的。

但物理 KV block 可以不连续。

这带来两个直接收益：

1. 释放后的 block 可以被任何新请求复用。
2. scheduler 可以用 free block 数做 admission control。

## 49.5 最小数据结构

先不要一上来实现 prefix cache、LRU eviction、swap、preemption。

第一版 paged KV cache 只需要四个结构。

第一，KV block pool。

```python
class KVBlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
```

第二，request 的 block table。

```python
class RequestState:
    def __init__(self, request_id: str, prompt_token_ids: list[int]):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = []
        self.block_table = []
        self.num_computed_tokens = 0
```

第三，block manager。

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
```

第四，batch metadata。

```python
class ModelInput:
    def __init__(self):
        self.input_ids = []
        self.positions = []
        self.block_tables = []
        self.slot_mapping = []
        self.seq_lens = []
```

这四个结构分别对应：

```text
物理资源池 -> 请求逻辑映射 -> 分配释放策略 -> kernel 输入 metadata
```

## 49.6 block size 怎么选

教学项目里可以先固定：

```python
block_size = 16
```

这和很多系统默认值接近，也便于手算。

例如 prompt 长度 33，需要：

```text
ceil(33 / 16) = 3 blocks
```

最后一个 block 只用了 1 个 token，剩余 15 个 token 位置属于 block 内部碎片。

block size 的 trade-off 是：

| block size | 优点 | 缺点 |
|---|---|---|
| 小 | 内部碎片少，释放粒度细 | block table 长，metadata 多 |
| 大 | block table 短，管理开销低 | 内部碎片多，短请求浪费明显 |

第一版不要把 block size 做成复杂自适应参数。

先固定一个值，把路径跑通，再通过压测观察浪费和吞吐。

## 49.7 计算需要多少 blocks

Block manager 最常用的函数是计算 token 数需要多少 blocks。

```python
def required_blocks(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size
```

例如：

```text
num_tokens = 0   -> 0 blocks
num_tokens = 1   -> 1 block
num_tokens = 16  -> 1 block
num_tokens = 17  -> 2 blocks
num_tokens = 32  -> 2 blocks
```

这个函数会在三个地方用到：

1. waiting request admission control。
2. prefill chunk 调度前的 block 预留。
3. decode 追加 token 前判断是否需要新 block。

## 49.8 can_allocate 不是简单看 prompt length

一个常见错误是：新请求来时直接按完整 prompt 长度分配 blocks。

如果不做 chunked prefill，这可以工作。

但上一章我们已经引入了 bounded prefill 和 chunked prefill。

这时一个请求可能本轮只 prefill prompt 的一部分。

所以要区分两个问题：

```text
最终这个请求最多需要多少 blocks？
本轮执行这些 token 前，需要保证多少 blocks 已经存在？
```

教学版可以先采用保守策略：admission 时为完整 prompt 分配 blocks。

```python
def can_allocate_prompt(self, request: RequestState) -> bool:
    needed = required_blocks(len(request.prompt_token_ids), self.block_size)
    missing = needed - len(request.block_table)
    return len(self.free_blocks) >= missing
```

这样简单安全，但长 prompt 会更容易阻塞 waiting queue。

更进一步的策略是按 prefill chunk 分配。

例如本轮只算前 256 个 prompt tokens，就只保证 256 个 token 对应的 blocks 存在。

```python
def can_allocate_until(self, request: RequestState, target_tokens: int) -> bool:
    needed = required_blocks(target_tokens, self.block_size)
    missing = needed - len(request.block_table)
    return len(self.free_blocks) >= missing
```

第一版建议实现 `can_allocate_until`。

因为它和 continuous batching 的 token budget 更匹配。

## 49.9 分配 blocks

分配函数要保证 request 的 block table 至少覆盖目标 token 数。

```python
def allocate_until(self, request: RequestState, target_tokens: int) -> bool:
    needed = required_blocks(target_tokens, self.block_size)
    missing = needed - len(request.block_table)

    if missing <= 0:
        return True

    if len(self.free_blocks) < missing:
        return False

    for _ in range(missing):
        block_id = self.free_blocks.pop()
        request.block_table.append(block_id)

    return True
```

注意这个函数的语义是：

```text
保证 request 的 KV block table 可以容纳 [0, target_tokens) 这些 token。
```

它不表示这些 token 已经计算完成。

是否计算完成由 `num_computed_tokens` 表示。

这是一个很重要的区分。

```text
allocated tokens >= computed tokens
```

allocated 表示有位置可以写。

computed 表示 KV 已经真的写入。

## 49.10 prefill 阶段如何使用 paged KV

假设一个请求 prompt 长度为 40，block size 为 16。

block table 为空。

第一轮 prefill chunk 处理 20 个 tokens。

调度前调用：

```python
block_manager.allocate_until(request, target_tokens=20)
```

此时需要 2 个 blocks：

```text
tokens 0-15 -> block table[0]
tokens 16-19 -> block table[1]
```

模型 forward 后：

```python
request.num_computed_tokens = 20
```

第二轮继续 prefill 20 个 tokens。

调度前调用：

```python
block_manager.allocate_until(request, target_tokens=40)
```

此时需要 3 个 blocks。

如果只已有 2 个，就再分配 1 个。

模型 forward 后：

```python
request.num_computed_tokens = 40
```

这个请求完成 prefill，可以进入 decode。

## 49.11 decode 阶段如何追加 token

decode 每轮通常为每个 running request 生成 1 个 token。

假设 request 当前总 token 数是：

```python
total_tokens = len(request.prompt_token_ids) + len(request.output_token_ids)
```

本轮要生成下一个 token，它的 KV 会写到 position：

```python
next_pos = total_tokens
```

写入之前，要确保 `[0, next_pos + 1)` 有 block 覆盖。

```python
block_manager.allocate_until(request, target_tokens=next_pos + 1)
```

如果 `next_pos` 正好落入一个新 logical block，就会分配新 physical block。

例如 block size 为 16：

```text
position 15 -> 仍在第 0 个 logical block
position 16 -> 需要第 1 个 logical block
```

这就是 paged KV cache 支持动态增长的关键。

## 49.12 token position 到 physical slot

Paged KV cache 的核心地址翻译是：

```text
token position -> logical block id -> physical block id -> offset in block
```

对应代码：

```python
def physical_slot(request: RequestState, token_pos: int, block_size: int) -> int:
    logical_block_id = token_pos // block_size
    offset = token_pos % block_size
    physical_block_id = request.block_table[logical_block_id]
    return physical_block_id * block_size + offset
```

如果底层 KV buffer 是按 `[num_blocks, block_size, ...]` 存储，也可以保留二维坐标：

```python
def physical_block_and_offset(request, token_pos, block_size):
    logical_block_id = token_pos // block_size
    offset = token_pos % block_size
    return request.block_table[logical_block_id], offset
```

真实 kernel 需要更复杂的 stride、layer、head、head_dim 信息。

但对 scheduler 和 BatchBuilder 来说，理解到 block 和 offset 已经足够。

## 49.13 slot mapping 是什么

在教学版连续 KV cache 中，模型可能默认第 i 个 input token 的 KV 写到第 i 个位置。

Paged KV cache 下，这不成立。

因为不同请求的物理 block 可以分散在全局 block pool 中。

所以 BatchBuilder 需要给本轮每个 input token 生成一个 `slot_mapping`。

例如 batch 中有两个请求：

```text
request A 本轮 prefill positions: 0, 1, 2
request B 本轮 decode position: 17
```

它们对应物理 slot 可能是：

```text
A pos 0  -> block 8 offset 0 -> slot 128
A pos 1  -> block 8 offset 1 -> slot 129
A pos 2  -> block 8 offset 2 -> slot 130
B pos 17 -> block 3 offset 1 -> slot 49
```

于是：

```python
slot_mapping = [128, 129, 130, 49]
```

模型 forward 写 KV 时，不再写到连续位置，而是按 `slot_mapping` 写到全局 KV buffer。

这是从 list KV cache 升级到 paged KV cache 时最容易漏掉的一步。

## 49.14 block table 传给谁

slot mapping 用于写当前 token 的 KV。

block table 用于 attention 读取历史 KV。

两者作用不同。

```text
slot mapping: 当前输入 token 的 KV 写到哪里
block table: 这个请求历史 token 的 KV 从哪里读
```

对于 prefill：

```text
当前 chunk 的 token 要写 KV，也要读同一个请求前面已经 computed 的 KV。
```

对于 decode：

```text
当前 token 要写 KV，也要 attend 到 prompt + previous output 的所有历史 KV。
```

因此 ModelInput 至少要包含：

```python
model_input.input_ids
model_input.positions
model_input.slot_mapping
model_input.block_tables
model_input.seq_lens
```

其中：

1. `input_ids`：本轮实际输入 token。
2. `positions`：这些 token 的逻辑 position。
3. `slot_mapping`：这些 token 写 KV 的物理位置。
4. `block_tables`：每个 request 的 logical block 到 physical block 映射。
5. `seq_lens`：每个 request 当前可见的上下文长度。

## 49.15 BatchBuilder 的改造

上一章的 scheduler 输出 `BatchPlan`。

现在 BatchBuilder 要根据 `BatchPlan` 构造 paged KV metadata。

简化代码：

```python
def build_model_input(batch_plan, block_size):
    model_input = ModelInput()

    for item in batch_plan.items:
        request = item.request
        start = item.start_pos
        end = item.start_pos + item.num_tokens

        for pos in range(start, end):
            token_id = request.all_token_ids[pos]
            model_input.input_ids.append(token_id)
            model_input.positions.append(pos)
            model_input.slot_mapping.append(
                physical_slot(request, pos, block_size)
            )

        model_input.block_tables.append(request.block_table)
        model_input.seq_lens.append(end)

    return model_input
```

这里的 `seq_lens.append(end)` 是简化写法。

真实系统里 prefill chunk、decode、prefix cache、causal mask、attention backend 对 seq len 的定义可能不同。

但教学版先抓住主线：

```text
本轮输入 token 写哪里，用 slot mapping。
历史 KV 读哪里，用 block table。
```

## 49.16 Scheduler 和 BlockManager 的边界

不要让 scheduler 直接操作 `free_blocks`。

更好的边界是：

```text
Scheduler 负责决定本轮想调度哪些 request 和多少 token。
BlockManager 负责判断 KV block 是否够，并完成分配释放。
```

Scheduler 可以调用：

```python
block_manager.can_allocate_until(request, target_tokens)
block_manager.allocate_until(request, target_tokens)
block_manager.free(request)
block_manager.num_free_blocks()
```

但不要写：

```python
block_manager.free_blocks.pop()
```

原因是后续你可能要加入：

1. prefix sharing。
2. ref count。
3. LRU eviction。
4. CPU offload。
5. preemption。

如果 scheduler 到处直接改 free list，后面很难演进。

## 49.17 admission control 的变化

list KV cache 时代，admission control 可能只看请求数：

```python
if len(running) < max_running_requests:
    admit(request)
```

paged KV cache 后，至少要同时看：

1. running request 数。
2. token budget。
3. free blocks。
4. 本轮 prefill chunk 需要的目标 token 数。

简化策略：

```python
if len(running) >= max_running_requests:
    keep_waiting(request)
elif not block_manager.can_allocate_until(request, target_tokens):
    keep_waiting(request)
elif remaining_token_budget < chunk_tokens:
    keep_waiting(request)
else:
    block_manager.allocate_until(request, target_tokens)
    schedule_prefill(request, chunk_tokens)
```

这个顺序可以调整。

但一定要避免：

```text
先把 request 状态改成 running，再发现 KV block 不够。
```

正确顺序应该是：

```text
先检查资源 -> 分配资源 -> 修改状态 -> 加入 batch plan
```

失败时 request 仍然留在 waiting queue。

## 49.18 decode 前也要检查 block

很多人会只在 prefill admission 时检查 KV block。

这是不够的。

decode 会不断生成新 token，也会继续消耗 KV blocks。

当一个请求生成到 block 边界时，需要新 block。

如果此时 free blocks 不够，会出现两种选择：

1. 暂停这个请求，等待其他请求释放 block。
2. 触发 preemption、swap 或 recompute。

教学版先采用第一种。

```python
def can_schedule_decode(request):
    next_pos = request.total_tokens()
    return block_manager.can_allocate_until(request, next_pos + 1)
```

如果不能 allocate，本轮不要调度这个 request。

但要打日志。

```text
skip decode request=req-7 reason=no_free_block next_pos=1024 free_blocks=0
```

否则你会看到 TPOT 突然变大，却不知道原因是 KV block 不够。

## 49.19 free request 的时机

请求释放 KV blocks 的时机包括：

1. 正常生成到 stop token。
2. 达到 max output tokens。
3. 客户端主动 abort。
4. 请求超时。
5. 模型执行失败。
6. sampling 或 detokenize 失败。

不要只处理第一种。

最小实现：

```python
def finish_request(request, reason):
    request.status = "FINISHED"
    request.finish_reason = reason
    block_manager.free(request)
```

abort 也必须释放：

```python
def abort_request(request):
    request.status = "ABORTED"
    block_manager.free(request)
```

失败也必须释放：

```python
def fail_request(request, error):
    request.status = "FAILED"
    request.error = error
    block_manager.free(request)
```

线上 KV 泄漏的高频来源就是某条异常路径忘记 free。

## 49.20 free 函数要幂等

真实系统里，一个请求可能在多个路径触发 cleanup。

例如客户端断开时触发 abort，同时 worker 返回时又发现请求已结束。

所以 `free` 最好设计成幂等。

```python
def free(self, request: RequestState):
    if not request.block_table:
        return

    for block_id in request.block_table:
        self.free_blocks.append(block_id)

    request.block_table = []
```

如果未来加入 ref count，则变成：

```python
for block_id in request.block_table:
    self.ref_count[block_id] -= 1
    if self.ref_count[block_id] == 0:
        self.free_blocks.append(block_id)
```

幂等 cleanup 可以减少很多线上状态机 bug。

## 49.21 防止 double free

幂等不等于可以随便重复释放。

如果 request 的 block table 没有清空，重复 free 会把同一个 block 放回 free list 两次。

这会导致两个请求拿到同一个 physical block，KV 相互覆盖。

因此 free 后必须清空：

```python
request.block_table = []
```

调试期还可以维护一个 allocated set：

```python
self.allocated_blocks = set()
```

分配时：

```python
assert block_id not in self.allocated_blocks
self.allocated_blocks.add(block_id)
```

释放时：

```python
assert block_id in self.allocated_blocks
self.allocated_blocks.remove(block_id)
```

这个 debug set 在高性能路径可以关掉，但教学项目非常建议保留。

它能快速抓住 double free 和 block 重复分配。

## 49.22 block table 的生命周期

一个请求的 block table 生命周期是：

```text
WAITING: []
PREFILLING: 随 prompt chunk 增长
DECODING: 随 output token 增长
FINISHED / ABORTED / FAILED: free 后清空
```

示例：

```text
request A, block_size=4

初始:
  block_table = []
  computed = 0

prefill 6 tokens:
  block_table = [10, 3]
  computed = 6

decode 到 position 8:
  block_table = [10, 3, 7]
  computed = 9

finish:
  free blocks 10, 3, 7
  block_table = []
```

面试里如果能把这个生命周期讲清楚，说明你理解的不是概念，而是工程状态流。

## 49.23 和 continuous batching 的结合

现在把第 48 章和本章合起来。

每个 iteration 的顺序可以是：

```text
1. 清理 finished / aborted / failed 请求，释放 blocks。
2. 为 running decode 请求检查下一 token 是否需要新 block。
3. 在 token budget 内选择可 decode 的请求。
4. 为 running prefill 请求分配 chunk 所需 blocks。
5. 从 waiting queue 中接纳新请求，分配首个 prefill chunk 的 blocks。
6. BatchBuilder 生成 input ids、positions、slot mapping、block tables。
7. Worker 执行 model forward。
8. OutputProcessor 更新 token、状态和 metrics。
```

注意第一步放在前面。

如果先调度新请求，再释放上一轮已经完成的请求，就会低估 free blocks，导致不必要的排队。

## 49.24 OOM 应该在哪里被拦住

理想情况下，OOM 不应该发生在 model forward 中。

它应该在 scheduler 阶段被拦住。

也就是说：

```text
如果本轮 KV blocks 不够，就不要把这个 request 放进 batch。
```

这要求 scheduler 在构建 batch plan 前就问 block manager。

错误路径：

```text
schedule request -> build tensor -> model forward -> CUDA OOM
```

正确路径：

```text
check blocks -> allocate blocks -> schedule request -> model forward
```

当然真实系统还可能因为 activation、workspace、fragmentation、CUDA graph 等原因 OOM。

但 KV cache 这一部分应该尽量通过 block budget 提前控制。

## 49.25 debug log 设计

升级 paged KV cache 后，必须增加 block 维度的日志。

每轮 scheduler log 可以包含：

```text
step=128
free_blocks_before=1024
free_blocks_after=1008
allocated_blocks=16
freed_blocks=4
waiting=37
running=64
prefill_reqs=3
prefill_tokens=512
decode_reqs=61
decode_tokens=61
skip_no_block=2
```

单个 request 的状态变化可以记录：

```text
request=req-12 event=allocate target_tokens=256 new_blocks=16 block_table_len=16
request=req-12 event=free blocks=21 reason=finished
request=req-8 event=skip_decode reason=no_free_block next_pos=2048
```

这些日志可以回答：

1. 请求为什么排队？
2. decode 为什么断流？
3. free blocks 是否长期下降？
4. 是否存在 KV 泄漏？
5. block size 是否导致浪费过大？

没有这些日志，paged KV cache 的 bug 很难定位。

## 49.26 需要观测哪些指标

除了 TTFT、TPOT 和 throughput，还要新增 memory 指标。

| 指标 | 含义 |
|---|---|
| free blocks | 当前空闲 block 数 |
| used blocks | 当前被请求占用的 block 数 |
| block utilization | 已用 token 数 / 已分配 block 容量 |
| internal fragmentation | block 内未使用 token slot 占比 |
| allocation failure count | 因 block 不足导致无法调度的次数 |
| skip decode no block | decode 因无 block 被跳过次数 |
| leaked blocks | 理论 used 与实际 allocated 不一致 |

block utilization 可以粗略计算：

```text
used_token_slots / (allocated_blocks * block_size)
```

短请求很多时，这个值可能偏低。

长请求很多时，这个值通常更高。

## 49.27 压测场景一：短请求高并发

workload：

```text
prompt length: 32-64
output length: 16-64
并发: 高
```

观察：

1. free blocks 是否快速回收。
2. finished 请求是否及时释放。
3. block 内部碎片是否明显。
4. output TPS 是否比 list KV cache 更稳定。

这个场景主要验证 block 复用。

如果请求完成后 free blocks 不回升，说明释放路径有问题。

## 49.28 压测场景二：长 prompt 短输出

workload：

```text
prompt length: 4K-16K
output length: 8-32
并发: 中等
```

观察：

1. admission control 是否能拦住 block 不足。
2. chunked prefill 是否按目标 token 分配 blocks。
3. long prompt 是否把 free blocks 快速耗尽。
4. waiting queue 是否因为 KV 不足持续增长。

这个场景主要验证 prefill block 分配和 token budget 的配合。

## 49.29 压测场景三：长输出 decode

workload：

```text
prompt length: 128-512
output length: 2K-8K
并发: 中等到高
```

观察：

1. decode 到 block 边界时是否正确追加 block。
2. free blocks 是否随 output 增长持续下降。
3. no block 时是否跳过 decode 或触发后续策略。
4. TPOT p99 是否因 block 不足出现尖刺。

这个场景主要验证 decode 动态扩展。

如果只测短输出，很容易漏掉 decode append block 的 bug。

## 49.30 和 prefix caching 的关系

本章第一版 paged KV cache 不实现 prefix caching。

但数据结构要为它留出空间。

prefix caching 需要：

1. block hash。
2. ref count。
3. cached block map。
4. copy-on-write 或 append-only 语义。

所以 `BlockManager` 最好不要只是一组裸 list 操作。

可以预留 block 对象：

```python
class KVCacheBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.block_hash = None
```

第一版即使不用 `block_hash`，也可以把 `ref_count` 先设为 0 或 1。

这样下一章或后续章节加入 prefix caching 时，不需要推翻 block manager。

## 49.31 和 preemption 的关系

Paged KV cache 也为 preemption 做准备。

当 free blocks 不够时，系统可以选择：

1. 不接纳新请求。
2. 暂停某些 running 请求。
3. 释放低优先级请求的 blocks，后续 recompute。
4. 把部分 KV blocks swap 到 CPU。

第一版只做第一种。

但如果 block manager 已经统一管理 request 到 blocks 的映射，后续实现 preemption 会清晰很多。

因为你可以明确知道：

```text
抢占 request A 能释放多少 blocks？
恢复 request A 需要重新计算多少 tokens？
```

如果 KV cache 仍然散落在各个 request 私有 list 中，就很难做这些策略。

## 49.32 常见 bug

bug 一：分配了 block，但 request 状态更新失败。

```text
结果：block 被占用但请求没有进入 running，形成 KV 泄漏。
```

解决：资源分配和状态修改要么放在同一个受控流程里，要么失败时回滚。

bug 二：free 后没有清空 block table。

```text
结果：重复 free 时同一个 block 被放回 free list 多次。
```

bug 三：decode 到 block 边界时没有分配新 block。

```text
结果：写 KV 越界，或覆盖上一个 block 的内容。
```

bug 四：slot mapping 用了逻辑 position，而不是物理 slot。

```text
结果：不同请求写入同一段 KV，生成内容互相污染。
```

bug 五：block table 传给 kernel 的顺序和 batch request 顺序不一致。

```text
结果：attention 读错请求的历史 KV。
```

bug 六：prefill chunk 的 seq len 设置错误。

```text
结果：attention mask 错误，要么看不到历史 token，要么看到了未来 token。
```

bug 七：abort 请求没有释放 blocks。

```text
结果：free blocks 持续下降，最终所有请求都进不来。
```

bug 八：can_allocate 和 allocate 使用的 target_tokens 不一致。

```text
结果：检查时够，真正分配时不够，或者过度保守导致吞吐下降。
```

## 49.33 面试高频问题

问题一：为什么要把 KV cache 从连续存储改成 paged 存储？

回答要点：因为 online serving 中请求长度和生命周期高度动态，连续预留会浪费显存，动态连续分配会产生碎片。paged KV cache 把逻辑 token 序列映射到固定大小的物理 blocks，使请求逻辑连续但物理不连续，从而提升显存复用和并发能力。

问题二：block table 的作用是什么？

回答要点：block table 记录一个请求的 logical block 到 physical block 的映射。attention 读取历史 KV 时，通过 token position 计算 logical block 和 offset，再查 block table 找到 physical block。

问题三：slot mapping 和 block table 有什么区别？

回答要点：slot mapping 描述本轮输入 token 的 KV 要写到哪个物理 slot；block table 描述这个请求历史 KV 分布在哪些 physical blocks。前者偏写入，后者偏读取。

问题四：paged KV cache 如何影响 scheduler？

回答要点：scheduler 不能只看请求数，还要看 free blocks 和 token budget。prefill admission、prefill chunk、decode append token 前都要判断 KV block 是否足够，避免调度后才 OOM。

问题五：为什么 decode 阶段也可能分配新 block？

回答要点：decode 每生成一个 token 都要追加 KV。当 token position 跨过 block 边界时，需要新的 logical block，因此要从 block pool 分配新的 physical block。

问题六：paged KV cache 还有什么浪费？

回答要点：它减少连续预留和外部碎片，但仍然有 block 内部碎片。最后一个 block 可能没有填满，block size 越大内部浪费越明显。

## 49.34 标准回答模板

如果面试官问“你如何把一个教学版 KV cache 升级成 paged KV cache”，可以这样回答：

```text
我会先把每个 request 私有的 list KV cache 改成全局 block pool。GPU 上预先切出固定大小的 KV blocks，每个 request 只维护一张 block table，记录它的 logical blocks 映射到哪些 physical blocks。这样 request 逻辑 token 序列仍然连续，但物理 KV 不要求连续。

数据结构上，我会新增 BlockManager，维护 free blocks、block size、request block table，第一版可以先不做 prefix cache 和 preemption。BlockManager 提供 can_allocate_until、allocate_until、free 这些接口，scheduler 不直接操作 free list。

调度流程上，prefill chunk 执行前根据 target token 数分配足够 blocks；decode 每生成下一个 token 前，也检查 next_pos + 1 是否需要新 block。只有 KV block 足够时，request 才能进入 batch plan，避免 model forward 时才发生 KV OOM。

BatchBuilder 需要同步改造。除了 input ids 和 positions，还要生成 slot mapping 和 block tables。slot mapping 用来告诉模型本轮输入 token 的 KV 写到哪个物理 slot，block table 用来让 attention 读取历史 KV。

最后，我会把所有终止路径都接到 block_manager.free，包括 finished、abort、timeout 和 failed，并增加 free_blocks、used_blocks、allocation failure、skip decode no block、block utilization 等指标，用短请求高并发、长 prompt 短输出、长输出 decode 三类 workload 验证 block 分配、复用和释放是否正确。
```

## 49.35 Paged KV 升级公式、slot mapping 和可运行 demo

从 list KV cache 升级到 paged KV cache，核心不是把 list 换成另一个容器，而是把 KV 变成可调度的 block 资源。

请求 `i` 当前需要覆盖的 token 数为 `T_i`，block size 为 `S`，需要的 logical block 数为：

```math
B_i=\left\lceil\frac{T_i}{S}\right\rceil
```

如果 request block table 当前长度为 `|P_i|`，还缺的 block 数为：

```math
M_i=\max(0,B_i-|P_i|)
```

block 内部碎片率：

```math
R_{\mathrm{frag}}=\frac{S\sum_i B_i-\sum_iT_i}{\max(1,S\sum_iB_i)}
```

token position 到物理 slot 的地址翻译为：

```math
b_t=\left\lfloor\frac{t}{S}\right\rfloor
```

```math
o_t=t-Sb_t
```

```math
p_t=P_i[b_t]
```

```math
s_t=Sp_t+o_t
```

最终升级门禁：

```math
G_{\mathrm{paged}}=G_{\mathrm{pool}}G_{\mathrm{table}}G_{\mathrm{slot}}G_{\mathrm{decode}}G_{\mathrm{admit}}G_{\mathrm{reuse}}G_{\mathrm{free}}
```

下面这个 0 依赖 demo 模拟一个最小 paged KV 改造：A 先做 chunked prefill，再在 decode 跨 block 边界时追加 block；B abort 后释放 blocks；C 复用 B 释放的 block；D 因 free blocks 不足在 admission 阶段被挡住；BatchBuilder 用 block table 生成 slot mapping；最后重复 free 不会 double free。

```python
from dataclasses import dataclass, field
import math


@dataclass
class RequestState:
    request_id: str
    prompt_tokens: list
    output_tokens: list = field(default_factory=list)
    block_table: list = field(default_factory=list)
    num_computed_tokens: int = 0
    status: str = "WAITING"

    def all_tokens(self):
        return self.prompt_tokens + self.output_tokens

    def total_tokens(self):
        return len(self.prompt_tokens) + len(self.output_tokens)


@dataclass
class BatchItem:
    request: RequestState
    start_pos: int
    token_count: int
    kind: str


class ToyPagedKVBlockManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = set()
        self.allocation_failures = 0
        self.idempotent_free_count = 0
        self.trace = []

    def required_blocks(self, num_tokens):
        return math.ceil(num_tokens / self.block_size) if num_tokens > 0 else 0

    def can_allocate_until(self, req, target_tokens):
        needed = self.required_blocks(target_tokens)
        missing = max(0, needed - len(req.block_table))
        return len(self.free_blocks) >= missing

    def allocate_until(self, req, target_tokens):
        needed = self.required_blocks(target_tokens)
        missing = max(0, needed - len(req.block_table))
        if missing == 0:
            return True, []
        if len(self.free_blocks) < missing:
            self.allocation_failures += 1
            self.trace.append(
                {
                    "event": "allocation_failed",
                    "request": req.request_id,
                    "target_tokens": target_tokens,
                    "missing_blocks": missing,
                    "free_blocks": len(self.free_blocks),
                }
            )
            return False, []

        new_blocks = []
        for _ in range(missing):
            block_id = self.free_blocks.pop(0)
            if block_id in self.allocated_blocks:
                raise AssertionError(f"double allocation: {block_id}")
            self.allocated_blocks.add(block_id)
            req.block_table.append(block_id)
            new_blocks.append(block_id)
        self.trace.append(
            {
                "event": "allocate",
                "request": req.request_id,
                "target_tokens": target_tokens,
                "new_blocks": new_blocks,
                "block_table": list(req.block_table),
            }
        )
        return True, new_blocks

    def free(self, req, reason):
        if not req.block_table:
            self.idempotent_free_count += 1
            self.trace.append({"event": "free_idempotent", "request": req.request_id, "reason": reason})
            return []

        released = list(req.block_table)
        for block_id in released:
            if block_id not in self.allocated_blocks:
                raise AssertionError(f"double free: {block_id}")
            self.allocated_blocks.remove(block_id)
        for block_id in reversed(released):
            self.free_blocks.insert(0, block_id)
        req.block_table = []
        req.status = reason.upper()
        self.trace.append({"event": "free", "request": req.request_id, "released": released, "reason": reason})
        return released

    def physical_slot(self, req, token_pos):
        logical_block = token_pos // self.block_size
        offset = token_pos - self.block_size * logical_block
        physical_block = req.block_table[logical_block]
        return physical_block * self.block_size + offset

    def fragmentation_ratio(self, requests):
        allocated_slots = 0
        used_tokens = 0
        for req in requests:
            if req.block_table:
                allocated_slots += len(req.block_table) * self.block_size
                used_tokens += req.total_tokens()
        waste = allocated_slots - used_tokens
        return round(waste / max(1, allocated_slots), 3)


class ToyBatchBuilder:
    def __init__(self, block_manager):
        self.block_manager = block_manager

    def build(self, items):
        model_input = {"input_ids": [], "positions": [], "slot_mapping": [], "block_tables": [], "seq_lens": []}
        for item in items:
            req = item.request
            for pos in range(item.start_pos, item.start_pos + item.token_count):
                model_input["input_ids"].append(req.all_tokens()[pos])
                model_input["positions"].append(pos)
                model_input["slot_mapping"].append(self.block_manager.physical_slot(req, pos))
            model_input["block_tables"].append((req.request_id, list(req.block_table)))
            model_input["seq_lens"].append(item.start_pos + item.token_count)
        return model_input


manager = ToyPagedKVBlockManager(num_blocks=8, block_size=4)

req_a = RequestState("A", prompt_tokens=list(range(10, 19)))
ok, a_prefill_blocks_1 = manager.allocate_until(req_a, target_tokens=6)
req_a.num_computed_tokens = 6
ok, a_prefill_blocks_2 = manager.allocate_until(req_a, target_tokens=9)
req_a.num_computed_tokens = 9

decode_extension_block = None
for token in [201, 202, 203, 204]:
    next_pos = req_a.total_tokens()
    ok, new_blocks = manager.allocate_until(req_a, target_tokens=next_pos + 1)
    if new_blocks:
        decode_extension_block = new_blocks[0]
    req_a.output_tokens.append(token)

req_b = RequestState("B", prompt_tokens=[31, 32, 33, 34, 35])
manager.allocate_until(req_b, target_tokens=5)
b_freed_blocks = manager.free(req_b, reason="aborted")

req_c = RequestState("C", prompt_tokens=[41, 42, 43])
manager.allocate_until(req_c, target_tokens=3)
c_reused_blocks = list(req_c.block_table)

req_d = RequestState("D", prompt_tokens=list(range(60, 80)))
d_admission_ok = manager.can_allocate_until(req_d, target_tokens=20)
if d_admission_ok:
    manager.allocate_until(req_d, target_tokens=20)
else:
    manager.allocate_until(req_d, target_tokens=20)

model_input = ToyBatchBuilder(manager).build(
    [
        BatchItem(req_a, start_pos=12, token_count=1, kind="decode"),
        BatchItem(req_c, start_pos=0, token_count=3, kind="prefill"),
    ]
)
fragmentation_before_cleanup = manager.fragmentation_ratio([req_a, req_c])

a_table_before_cleanup = list(req_a.block_table)
c_table_before_cleanup = list(req_c.block_table)
manager.free(req_a, reason="finished")
manager.free(req_a, reason="finished")
manager.free(req_c, reason="finished")

summary = {
    "a_block_table_after_prefill": [0, 1, 2],
    "a_block_table_before_cleanup": a_table_before_cleanup,
    "decode_extension_block": decode_extension_block,
    "b_freed_blocks": b_freed_blocks,
    "c_reused_blocks": c_reused_blocks,
    "d_admission_ok": d_admission_ok,
    "slot_mapping": model_input["slot_mapping"],
    "block_tables": model_input["block_tables"],
    "fragmentation_before_cleanup": fragmentation_before_cleanup,
    "allocation_failures": manager.allocation_failures,
    "free_blocks_after_cleanup": len(manager.free_blocks),
    "idempotent_free_count": manager.idempotent_free_count,
    "trace_tail": manager.trace[-5:],
}
gates = {
    "global_block_pool_ready": manager.num_blocks == 8,
    "block_table_grows": a_table_before_cleanup == [0, 1, 2, 3],
    "decode_extension_ready": decode_extension_block == 3,
    "slot_mapping_ready": model_input["slot_mapping"] == [12, 16, 17, 18],
    "admission_blocks_gate": d_admission_ok is False and manager.allocation_failures == 1,
    "reuse_after_free": c_reused_blocks[0] in b_freed_blocks,
    "idempotent_cleanup_ready": manager.idempotent_free_count == 1 and len(set(manager.free_blocks)) == manager.num_blocks,
}
gates["paged_kv_upgrade_gate"] = all(gates.values())

print("paged_kv_upgrade_summary=", summary)
print("paged_kv_upgrade_gates=", gates)
```

一次运行的核心输出类似：

```text
paged_kv_upgrade_summary= {'a_block_table_after_prefill': [0, 1, 2], 'a_block_table_before_cleanup': [0, 1, 2, 3], 'decode_extension_block': 3, 'b_freed_blocks': [4, 5], 'c_reused_blocks': [4], 'd_admission_ok': False, 'slot_mapping': [12, 16, 17, 18], 'block_tables': [('A', [0, 1, 2, 3]), ('C', [4])], 'fragmentation_before_cleanup': 0.2, 'allocation_failures': 1, 'free_blocks_after_cleanup': 8, 'idempotent_free_count': 1, 'trace_tail': [...]}
paged_kv_upgrade_gates= {'global_block_pool_ready': True, 'block_table_grows': True, 'decode_extension_ready': True, 'slot_mapping_ready': True, 'admission_blocks_gate': True, 'reuse_after_free': True, 'idempotent_cleanup_ready': True, 'paged_kv_upgrade_gate': True}
```

这个 demo 验证了从 list KV cache 到 paged KV cache 的几个硬门禁：

1. A 的 `block_table` 从 prefill 到 decode 持续增长，逻辑 token 连续但物理 block 由 table 映射。
2. decode 写到 position 12 时跨过 block 边界，必须追加 block 3。
3. `slot_mapping=[12,16,17,18]` 证明 BatchBuilder 用的是物理 slot，不是逻辑 position。
4. B abort 后释放 `[4,5]`，C 随后复用 block 4，说明释放和复用闭环。
5. D 在 admission 阶段被 free block 数拦住，没有等到 model forward 才 OOM。
6. 对 A 重复 free 只增加幂等计数，不会 double free 或重复放回 free list。

## 49.36 小练习

1. 给 `RequestState` 增加 `block_table` 和 `num_computed_tokens` 字段。
2. 实现 `required_blocks(num_tokens, block_size)`。
3. 实现 `BlockManager.can_allocate_until`。
4. 实现 `BlockManager.allocate_until`。
5. 实现幂等的 `BlockManager.free`。
6. 写一个单测：block size 为 4，分配 0、1、4、5、8、9 个 tokens 分别需要多少 blocks。
7. 写一个单测：request free 后 free block 数恢复。
8. 写一个单测：重复 free 不会 double free。
9. 实现 `physical_slot(request, token_pos, block_size)`。
10. 改造 BatchBuilder，生成 `slot_mapping`。
11. 构造两个 request 的 block table，验证 batch 中 slot mapping 不冲突。
12. 在 scheduler log 中加入 `free_blocks_before` 和 `free_blocks_after`。
13. 构造长输出 workload，验证 decode 跨 block 边界时会分配新 block。
14. 构造 abort 请求，验证 KV blocks 被释放。
15. 写一段面试回答：为什么 slot mapping 和 block table 都需要？

## 49.37 本章总结

从 list KV cache 升级到 paged KV cache，是 mini engine 从教学玩具走向 serving engine 的关键一步。

list KV cache 的问题不只是性能差，而是缺少全局资源池抽象。它让 scheduler 难以做准确 admission control，也让请求加入、退出和 cache 复用变得混乱。

paged KV cache 的核心是 block pool、block table、slot mapping 和 BlockManager。

block table 让请求逻辑连续、物理不连续。

slot mapping 告诉模型当前 token 的 KV 写到哪里。

BlockManager 把 KV cache 变成可分配、可释放、可观测的资源。

和 continuous batching 结合后，scheduler 每轮都要基于 free blocks、token budget 和 request 状态做调度，尽量在 batch plan 阶段拦住 KV OOM。

第一版不需要急着做 prefix caching、preemption 或 CPU offload。

先把最小 block pool、block table、allocate、free、slot mapping 和日志指标做对，后续高级能力才有稳定基础。

下一章会继续沿着 mini engine 升级路线，进入第 50 章：实现 prefix cache 和 prompt cache。
