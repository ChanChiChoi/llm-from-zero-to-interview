# 第 18 章 KV Cache Block Manager

上一章讲了 PagedAttention 的核心思想：把请求的逻辑 KV Cache 切成 logical blocks，并映射到 GPU 上的 physical blocks。本章继续讲谁来管理这些 blocks，这就是 KV Cache Block Manager。

如果说 PagedAttention 是“分页式读取 KV Cache 的 attention 机制”，那么 Block Manager 就是“负责分配、释放、复用、追踪 KV Cache block 的内存管理器”。

一句话概括：

> KV Cache Block Manager 把 GPU 显存中的 KV Cache blocks 当成可分配资源，负责为请求分配 block、维护 block table、在请求结束后回收 block，并支撑 prefix sharing 和 continuous batching。

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

## 18.15 常见误区

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

## 18.16 面试追问

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

## 18.17 小练习

1. 实现一个只有 `allocate` 和 `free` 的 BlockManager。
2. 假设 block size 为 16，prompt 1000 tokens，需要多少 blocks？
3. 模拟一个请求生成 token 时 block 不够，触发新 block 分配。
4. 模拟两个请求共享前三个 blocks，并用 ref_count 管理释放。
5. 设计 5 个 block manager 指标，用于排查线上 OOM。

## 18.18 本章小结

本章讲清了 KV Cache Block Manager。

PagedAttention 需要 physical KV blocks 和 block table，而 Block Manager 负责管理这些 blocks：分配、扩展、释放、引用计数、共享和资源预算。它把 KV Cache 从普通张量变成可调度资源，是 vLLM 支持高并发、continuous batching 和 prefix caching 的关键基础。

下一章我们会进入 Continuous Batching 与 iteration-level scheduling，讲 vLLM 如何每轮动态选择请求，让 GPU 持续处理有效工作。
