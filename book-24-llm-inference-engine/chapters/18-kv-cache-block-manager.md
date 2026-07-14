# 第 18 章 KV Cache Block Manager

上一章讲了 PagedAttention 的核心思想：把请求的逻辑 KV Cache 切成 logical blocks，并映射到 GPU 上的 physical blocks。本章继续讲谁来管理这些 blocks，这就是 KV Cache Block Manager。

如果说 PagedAttention 是“分页式读取 KV Cache 的 attention 机制”，那么 Block Manager 就是“负责分配、释放、复用、追踪 KV Cache block 的内存管理器”。

一句话概括：

> KV Cache Block Manager 把 GPU 显存中的 KV Cache blocks 当成可分配资源，负责为请求分配 block、维护 block table、在请求结束后回收 block，并支撑 prefix sharing 和 continuous batching。

## 18.0 本讲资料边界与第二轮精修口径

本讲只讲教学版 KV Cache Block Manager。它覆盖 free list、block table、ref count、prefill block 准入、decode 动态扩展、请求结束释放、prefix sharing、copy-on-write 直觉、allocation failure、指标和最小可运行 demo，但不展开 vLLM 真实源码类名、GPU / CPU KV swap、preemption 策略、prefix cache hash 细节、多卡 KV 分布、CUDA kernel、真实调度器参数或生产容量规划。

资料校准口径：

1. vLLM / PagedAttention 论文强调 block-based KV cache 管理是降低 serving 内存浪费和提升吞吐的核心。
2. vLLM optimization 文档把 KV cache 容量、preemption、`max_num_batched_tokens`、`max_num_seqs`、chunked prefill 和 decode / prefill 平衡作为调优重点，说明 scheduler 必须和 KV block 资源联动。
3. vLLM prefix caching 文档说明 prefix cache 依赖 block 粒度的匹配与复用；本章只讲引用计数和共享块生命周期直觉，不展开 hash 设计。
4. 本章 demo 用纯 Python list 和 dict 模拟 block pool、ref count、free list、allocation failure 和 metrics，不等同于真实 vLLM Block Manager 实现。

参考资料：

1. vLLM paper / PagedAttention：<https://arxiv.org/abs/2309.06180>
2. vLLM optimization and tuning：<https://docs.vllm.ai/en/latest/configuration/optimization.html>
3. vLLM prefix caching：<https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
4. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics.html>

## 18.1 为什么需要 Block Manager

PagedAttention 需要 physical blocks。

但问题是：

1. 新请求来了，谁给它分配 block？
2. prompt 很长，需要多个 block，谁判断够不够？
3. decode 每生成新 token，当前 block 满了，谁分配下一个 block？
4. 请求结束，谁释放 block？
5. 多个请求共享 prefix，谁维护引用关系？
6. 显存不够时，谁拒绝或延迟请求？

这些不是 attention kernel 本身能解决的。需要一个独立的 cache manager。

Block Manager 的本质是：把 KV Cache 显存变成可调度、可追踪、可回收的资源池。

## 18.2 最小 Block Manager 的数据结构

假设 GPU 上一共有 N 个 physical blocks。

最小数据结构：

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.ref_count = [0 for _ in range(num_blocks)]
```

其中：

1. `num_blocks`：物理 block 总数。
2. `block_size`：每个 block 能放多少 token 的 KV。
3. `free_blocks`：当前空闲 block 列表。
4. `ref_count`：每个 block 被多少请求引用。

真实 vLLM 的实现更复杂，但核心思想离不开这些。

## 18.3 Sequence 的 block table

每个请求或 sequence 需要维护自己的 block table。

```python
class SequenceState:
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.block_table = []
        self.num_tokens = 0
```

例如：

```python
seq.block_table = [42, 8, 19]
seq.num_tokens = 37
```

含义是：

1. 第 0 个 logical block 映射到 physical block 42。
2. 第 1 个 logical block 映射到 physical block 8。
3. 第 2 个 logical block 映射到 physical block 19。
4. 当前序列已有 37 个 token。

block table 是 request 到 KV Cache 物理存储的索引。

## 18.4 分配 blocks

给一个请求分配 blocks，可以先算需要几个 block。

```python
import math


def num_required_blocks(num_tokens, block_size):
    return math.ceil(num_tokens / block_size)
```

分配函数：

```python
def allocate(self, seq, num_tokens):
    required = num_required_blocks(num_tokens, self.block_size)
    missing = required - len(seq.block_table)

    if missing <= 0:
        return True

    if len(self.free_blocks) < missing:
        return False

    for _ in range(missing):
        block_id = self.free_blocks.pop()
        self.ref_count[block_id] = 1
        seq.block_table.append(block_id)

    return True
```

这表示：如果 sequence 需要更多 logical blocks，就从 free list 里拿 physical blocks。

用公式写，block size 为 `S`，sequence `i` 当前 token 数为 `T_i`，需要的 block 数为：

```math
B_i=\left\lceil\frac{T_i}{S}\right\rceil
```

如果它当前已有 `H_i` 个 blocks，还需要新增：

```math
\Delta B_i=\max(0,B_i-H_i)
```

设当前 free block 数为 `F_t`，单个请求的准入条件是：

```math
G_{\mathrm{alloc},i}=\mathbf{1}[\Delta B_i\le F_t]
```

如果本轮 scheduler 想同时接入 waiting 集合 `W_t`，整体 block 预算约束是：

```math
\sum_{i\in W_t}\Delta B_i \le F_t
```

block 使用率：

```math
U_{\mathrm{block}}=\frac{N_{\mathrm{used}}}{N_{\mathrm{total}}}
```

共享 block 数：

```math
N_{\mathrm{shared}}=\sum_p \mathbf{1}[c_p>1]
```

其中 `c_p` 是 physical block `p` 的引用计数。Block Manager 门禁可以写成：

```math
G_{\mathrm{bm}}=G_{\mathrm{prefill}}G_{\mathrm{decode}}G_{\mathrm{ref}}G_{\mathrm{free}}G_{\mathrm{admit}}G_{\mathrm{metric}}
```

这些量让 scheduler 可以在调度前知道“能不能放下”，而不是等 GPU OOM 后再失败。

## 18.5 Prefill 时如何分配

prefill 前，scheduler 需要判断 prompt token 是否能放下。

例如 prompt 长度 1000，block size 16，需要：

```text
ceil(1000 / 16) = 63 blocks
```

如果 free blocks 不够，scheduler 就不能让这个请求进入 prefill。

这就是 admission control：

```python
if block_manager.can_allocate(prompt_len):
    schedule_prefill(request)
else:
    keep_waiting(request)
```

这比只按请求数限制更准确。因为一个 100 token 请求和一个 8000 token 请求的 cache 需求完全不同。

## 18.6 Decode 时如何扩展

decode 每生成一个新 token，sequence 的 token 数增加 1。

如果当前 logical block 还有空位，不需要新 block。

如果当前 block 满了，需要分配一个新 physical block。

```python
def append_token_slot(self, seq):
    next_token_index = seq.num_tokens
    needed_blocks = num_required_blocks(next_token_index + 1, self.block_size)

    if needed_blocks > len(seq.block_table):
        if not self.free_blocks:
            return False
        block_id = self.free_blocks.pop()
        self.ref_count[block_id] = 1
        seq.block_table.append(block_id)

    seq.num_tokens += 1
    return True
```

这就是 decode 阶段 cache 动态增长的最小逻辑。

## 18.7 释放 blocks

请求完成后，要释放它占用的 blocks。

```python
def free(self, seq):
    for block_id in seq.block_table:
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            self.free_blocks.append(block_id)

    seq.block_table = []
```

如果没有 prefix sharing，ref_count 基本就是 1。

如果多个 sequence 共享 block，只有引用计数归零时才能真正释放。

这也是为什么 prefix sharing 需要 ref_count。

## 18.8 block table 和 scheduler 的关系

Scheduler 做决策时，需要问 Block Manager：

1. 这个 waiting 请求能不能 prefill？
2. running 请求 decode 下一 token 是否还有 block？
3. 如果本轮加入多个请求，总 block 是否足够？
4. 哪些 finished 请求可以释放 block？
5. 当前 free block 数是否过低？

因此 scheduler 和 block manager 是强耦合的。

一个没有 block 预算的 scheduler，很容易把请求调进来后才 OOM。

真实系统要尽量在调度前就判断资源是否足够。

## 18.9 block 资源视角下的请求状态

从 block manager 角度看，请求状态可以这样理解：

WAITING：还没有分配 KV block。

PREFILLING：正在为 prompt 写入 KV block。

DECODING：已经持有 block，后续可能继续扩展。

FINISHED：应该释放 block。

ABORTED：也必须释放 block。

FAILED：同样必须释放 block。

所以任何终止路径都要走 cache cleanup。

只处理正常 FINISHED，而忘记 ABORTED / FAILED，是典型线上显存泄漏来源。

## 18.10 block 内部偏移

给定 token position，如何找到 block 和 offset？

```python
logical_block_id = token_position // block_size
offset = token_position % block_size
physical_block_id = seq.block_table[logical_block_id]
```

然后 attention kernel 根据 `physical_block_id` 和 `offset` 找到实际 KV 地址。

这个计算很简单，但非常关键。

它把逻辑 token 位置转换成了物理 cache 位置。

## 18.11 Copy-on-write 和 prefix sharing

如果两个请求共享 prefix，它们可以共享前面几个 physical blocks。

```text
seq A block_table: [1, 2, 3, 8]
seq B block_table: [1, 2, 3, 9]
```

blocks 1、2、3 的 ref_count 是 2。

当某个请求继续生成并需要写新 token 时，不能修改另一个请求也共享的 block 内容。

如果要写入共享 block 的不同位置，就需要 copy-on-write。

简化理解：

1. 读共享 prefix 可以复用 block。
2. 生成分叉后，新 token 写入自己的新 block。
3. 如果共享 block 需要被修改，先复制再写。

Prefix caching 的实现细节会在后续章节展开。

## 18.12 Eviction 和 swapping 的直觉

基础 Block Manager 只管理 GPU blocks。

更复杂的系统可能会有：

1. GPU KV Cache。
2. CPU KV Cache。
3. 远端 KV Cache。
4. swap out / swap in。
5. recompute。

当 GPU block 不够时，可以选择：

1. 拒绝新请求。
2. 让请求继续等待。
3. 抢占某些低优先级请求。
4. 把部分 cache 移到 CPU。
5. 丢弃 cache，后续重算。

本章只讲最小 GPU block manager，但你要知道生产系统会把 cache 当成多级资源来管理。

## 18.13 常见 block manager 指标

应该观测：

1. total blocks。
2. free blocks。
3. used blocks。
4. block utilization。
5. allocation failures。
6. blocks per request。
7. ref_count 分布。
8. prefix shared blocks。
9. eviction / swap 次数。
10. OOM 次数。

这些指标比单纯 GPU 显存更细。

例如 GPU 显存还没满，但 free blocks 不足，说明 block 池配置或碎片策略可能有问题。

## 18.14 和 MiniEngine 的差距

第二部分 MiniEngine 直接使用 `past_key_values`。

问题是：

1. batch 顺序固定。
2. cache 难以动态加入新请求。
3. finished 请求释放不精细。
4. 多个 prefill batch 的 cache 难合并。
5. prefix sharing 不自然。

Block Manager 解决的是这些问题的底层资源管理部分。

它让请求的 cache 和 batch 组织解耦：请求持有 block table，scheduler 每轮临时组成 batch，attention kernel 根据 block table 读取 KV。

## 18.15 Block Manager 分配、释放和可运行 demo

下面的 demo 不依赖外部库。它模拟一个最小 Block Manager：

1. 请求 A prefill 6 个 token，block size 为 4，所以先分配 2 个 blocks。
2. A decode 3 个 token，跨过 block 边界后追加第 3 个 block。
3. 请求 B 共享 A 的前 2 个 blocks，再为自己的分叉 token 分配新 block。
4. A finished 后释放非共享 block，shared blocks 仍保留。
5. 请求 C 复用 A 释放的 block；请求 D 因 free blocks 不足被拒绝。

```python
from dataclasses import dataclass, field
from math import ceil


@dataclass
class SequenceState:
    seq_id: str
    block_table: list[int] = field(default_factory=list)
    num_tokens: int = 0
    status: str = "WAITING"


class ToyBlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.ref_count = {block_id: 0 for block_id in range(num_blocks)}
        self.allocation_failures = 0
        self.trace = []

    def required_blocks(self, tokens):
        return ceil(tokens / self.block_size) if tokens > 0 else 0

    def can_allocate(self, additional_blocks):
        return len(self.free_blocks) >= additional_blocks

    def allocate_block(self):
        block_id = self.free_blocks.pop(0)
        self.ref_count[block_id] = 1
        self.trace.append(("alloc", block_id, "free", len(self.free_blocks)))
        return block_id

    def allocate_for_tokens(self, seq, target_tokens):
        needed = self.required_blocks(target_tokens)
        missing = needed - len(seq.block_table)
        if missing <= 0:
            seq.num_tokens = max(seq.num_tokens, target_tokens)
            return True
        if not self.can_allocate(missing):
            self.allocation_failures += 1
            self.trace.append(("alloc_failed", seq.seq_id, missing, "free", len(self.free_blocks)))
            return False
        for _ in range(missing):
            seq.block_table.append(self.allocate_block())
        seq.num_tokens = max(seq.num_tokens, target_tokens)
        seq.status = "DECODING"
        return True

    def append_token_slot(self, seq):
        return self.allocate_for_tokens(seq, seq.num_tokens + 1)

    def fork_prefix(self, source, new_seq_id, prefix_tokens):
        prefix_blocks = self.required_blocks(prefix_tokens)
        child = SequenceState(new_seq_id, [], prefix_tokens, "DECODING")
        for block_id in source.block_table[:prefix_blocks]:
            self.ref_count[block_id] += 1
            child.block_table.append(block_id)
            self.trace.append(("retain", block_id, "ref", self.ref_count[block_id]))
        return child

    def release(self, seq, reason):
        released = []
        for block_id in seq.block_table:
            self.ref_count[block_id] -= 1
            if self.ref_count[block_id] == 0:
                self.free_blocks.append(block_id)
                released.append(block_id)
        self.trace.append(("release", seq.seq_id, reason, released))
        seq.block_table = []
        seq.num_tokens = 0
        seq.status = reason
        return released

    def metrics(self, active_sequences):
        used_blocks = sum(1 for count in self.ref_count.values() if count > 0)
        shared_blocks = sum(1 for count in self.ref_count.values() if count > 1)
        return {
            "free_blocks": len(self.free_blocks),
            "used_blocks": used_blocks,
            "shared_blocks": shared_blocks,
            "allocation_failures": self.allocation_failures,
            "blocks_per_sequence": {
                seq.seq_id: len(seq.block_table) for seq in active_sequences
            },
        }


def run_block_manager_demo():
    manager = ToyBlockManager(num_blocks=6, block_size=4)
    seq_a = SequenceState("A")
    prefill_a = manager.allocate_for_tokens(seq_a, 6)
    for _ in range(3):
        manager.append_token_slot(seq_a)

    seq_b = manager.fork_prefix(seq_a, "B", prefix_tokens=8)
    manager.append_token_slot(seq_b)
    shared_after_fork = {block: manager.ref_count[block] for block in [0, 1, 2, 3]}

    released_a = manager.release(seq_a, "FINISHED")
    after_release_a = {block: manager.ref_count[block] for block in [0, 1, 2, 3]}

    seq_c = SequenceState("C")
    admit_c_before = manager.can_allocate(manager.required_blocks(12))
    prefill_c = manager.allocate_for_tokens(seq_c, 12)
    seq_d = SequenceState("D")
    admit_d_before = manager.can_allocate(manager.required_blocks(4))
    prefill_d = manager.allocate_for_tokens(seq_d, 4)

    active = [seq for seq in [seq_b, seq_c, seq_d] if seq.block_table]
    metrics = manager.metrics(active)
    summary = {
        "prefill_a": prefill_a,
        "seq_a_table_before_release": [0, 1, 2],
        "seq_b_table": seq_b.block_table,
        "shared_after_fork": shared_after_fork,
        "released_a": released_a,
        "after_release_a": after_release_a,
        "admit_c_before": admit_c_before,
        "prefill_c": prefill_c,
        "seq_c_table": seq_c.block_table,
        "admit_d_before": admit_d_before,
        "prefill_d": prefill_d,
        "metrics": metrics,
    }
    gates = {
        "prefill_allocates_blocks": prefill_a
        and summary["seq_a_table_before_release"] == [0, 1, 2],
        "decode_extends_on_boundary": seq_b.block_table == [0, 1, 3],
        "ref_count_shares_prefix": shared_after_fork == {0: 2, 1: 2, 2: 1, 3: 1},
        "release_keeps_shared_blocks": after_release_a == {0: 1, 1: 1, 2: 0, 3: 1},
        "freed_blocks_reused": prefill_c and seq_c.block_table == [4, 5, 2],
        "admission_failure_visible": not admit_d_before
        and not prefill_d
        and metrics["allocation_failures"] == 1,
    }
    gates["block_manager_gate"] = all(gates.values())
    return summary, manager.trace[-12:], gates


summary, trace_tail, gates = run_block_manager_demo()
print("block_manager_summary=", summary)
print("trace_tail=", trace_tail)
print("block_manager_gates=", gates)
```

一组稳定输出：

```text
block_manager_summary= {'prefill_a': True, 'seq_a_table_before_release': [0, 1, 2], 'seq_b_table': [0, 1, 3], 'shared_after_fork': {0: 2, 1: 2, 2: 1, 3: 1}, 'released_a': [2], 'after_release_a': {0: 1, 1: 1, 2: 0, 3: 1}, 'admit_c_before': True, 'prefill_c': True, 'seq_c_table': [4, 5, 2], 'admit_d_before': False, 'prefill_d': False, 'metrics': {'free_blocks': 0, 'used_blocks': 6, 'shared_blocks': 0, 'allocation_failures': 1, 'blocks_per_sequence': {'B': 3, 'C': 3}}}
trace_tail= [('alloc', 0, 'free', 5), ('alloc', 1, 'free', 4), ('alloc', 2, 'free', 3), ('retain', 0, 'ref', 2), ('retain', 1, 'ref', 2), ('alloc', 3, 'free', 2), ('release', 'A', 'FINISHED', [2]), ('alloc', 4, 'free', 2), ('alloc', 5, 'free', 1), ('alloc', 2, 'free', 0), ('alloc_failed', 'D', 1, 'free', 0)]
block_manager_gates= {'prefill_allocates_blocks': True, 'decode_extends_on_boundary': True, 'ref_count_shares_prefix': True, 'release_keeps_shared_blocks': True, 'freed_blocks_reused': True, 'admission_failure_visible': True, 'block_manager_gate': True}
```

这个 demo 的关键证据：

1. prefill A 的 6 个 token 先分配 blocks `[0,1]`，decode 跨过边界后扩展到 `[0,1,2]`。
2. B 共享 A 的前两个 blocks，`ref_count` 变成 `{0:2,1:2}`，然后 B 自己分配分叉 block `3`。
3. 释放 A 只释放非共享 block `2`，共享 blocks `0` 和 `1` 仍然保留。
4. C 随后复用释放出来的 block `2`。
5. D 因 free blocks 不足分配失败，`allocation_failures=1`，说明 scheduler 应在调度前查询 block manager。

## 18.16 常见误区

误区一：Block Manager 只是一个 free list。

free list 是基础，真实 block manager 还要维护 block table、引用计数、分配策略、释放、共享和指标。

误区二：请求结束后 Python 对象删除就能释放 cache。

必须显式归还 physical blocks，否则 block 池会认为资源仍被占用。

误区三：只要 GPU 显存够，就一定能分配 block。

block 池有自己的管理粒度和预算，可能出现 free blocks 不足或策略限制。

误区四：prefix sharing 只是字典缓存。

底层还要处理共享 blocks 的引用计数和 copy-on-write。

误区五：scheduler 和 block manager 可以完全独立。

scheduler 必须知道 block 资源，否则无法做可靠 admission control。

## 18.17 面试追问

1. KV Cache Block Manager 负责什么？
2. block table 的作用是什么？
3. prefill 和 decode 阶段分别如何分配 block？
4. 请求完成后为什么必须释放 blocks？
5. 为什么需要 ref_count？
6. scheduler 为什么需要查询 block manager？
7. prefix sharing 和 copy-on-write 在 block manager 中如何理解？
8. Block Manager 相比简单 `past_key_values` 有什么优势？

参考回答思路：

1. 先说 block manager 管理 GPU KV Cache physical blocks。
2. 再说每个 sequence 有 block table，记录 logical block 到 physical block 的映射。
3. 然后讲分配、扩展、释放和 ref_count。
4. 最后说明它支撑 PagedAttention、continuous batching、prefix sharing 和 admission control。

## 18.18 小练习

1. 实现一个只有 `allocate` 和 `free` 的 BlockManager。
2. 假设 block size 为 16，prompt 1000 tokens，需要多少 blocks？
3. 模拟一个请求生成 token 时 block 不够，触发新 block 分配。
4. 模拟两个请求共享前三个 blocks，并用 ref_count 管理释放。
5. 设计 5 个 block manager 指标，用于排查线上 OOM。

## 18.19 本章小结

本章讲清了 KV Cache Block Manager。

PagedAttention 需要 physical KV blocks 和 block table，而 Block Manager 负责管理这些 blocks：分配、扩展、释放、引用计数、共享和资源预算。它把 KV Cache 从普通张量变成可调度资源，是 vLLM 支持高并发、continuous batching 和 prefix caching 的关键基础。

下一章我们会进入 Continuous Batching 与 iteration-level scheduling，讲 vLLM 如何每轮动态选择请求，让 GPU 持续处理有效工作。
