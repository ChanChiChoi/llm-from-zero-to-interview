# 第 17 章 PagedAttention 的核心思想

上一章讲了 vLLM 主要解决什么问题。本章进入 vLLM 最核心的机制：PagedAttention。

很多同学第一次听 PagedAttention，会以为它只是一个更快的 attention kernel。这个理解不完整。PagedAttention 的真正核心是：把 KV Cache 管理从“连续大块内存”变成“分页式块管理”，从而减少显存浪费，支持更高并发和更灵活的调度。

一句话概括：

> PagedAttention 借鉴操作系统分页思想，把请求的逻辑 KV Cache 切成固定大小的块，并映射到 GPU 上的物理 KV block，从而避免连续分配和显存碎片问题。

## 17.0 本讲资料边界与第二轮精修口径

本讲只解释 PagedAttention 的核心思想。它覆盖 logical block、physical block、block table、token position 到 KV block 的地址翻译、block size trade-off、block 内部浪费、请求释放后的 block 复用、prefix sharing / copy-on-write 直觉和最小可运行分页 KV demo，但不展开 vLLM 真实 CUDA kernel、真实 block manager 源码、attention kernel 性能调优、多卡 KV 分布、prefix cache 策略、swap / preemption 或生产参数配置。

资料校准口径：

1. vLLM / PagedAttention 论文把 KV cache 分成固定大小 blocks，并用 PagedAttention 让 attention kernel 能读取非连续物理 KV blocks，从而降低内存浪费并支持高吞吐 serving。
2. vLLM 文档和源码术语通常围绕 block table、KV cache block、scheduler、block manager、prefix caching 和 KV cache usage 展开；本章只抽象这些机制的稳定概念，不绑定具体类名或参数名。
3. 操作系统分页类比只用于帮助理解逻辑地址到物理地址的映射，不表示 GPU KV cache 完全等价于 CPU 虚拟内存。
4. 本章 demo 用纯 Python list 模拟 KV block storage、block table、ref count 和 free list，不实现真实 attention score 计算。

参考资料：

1. vLLM paper / PagedAttention：<https://arxiv.org/abs/2309.06180>
2. vLLM documentation：<https://docs.vllm.ai/>
3. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics.html>

## 17.1 问题从 KV Cache 开始

自回归生成需要 KV Cache。

每个请求的 cache 会随着上下文长度增长：

```text
prompt tokens + generated tokens -> KV Cache length
```

如果一个请求上下文很长，它需要大量 cache。如果很多请求同时运行，总 cache 占用会非常大。

传统做法可能为每个请求分配一段连续 cache 空间。

问题是请求长度不可预测：

1. 有的请求很短，很快结束。
2. 有的请求 prompt 很长。
3. 有的请求生成很多 token。
4. 有的请求中途取消。
5. batch 中请求不断加入和退出。

连续分配在这种动态环境里很容易浪费显存。

## 17.2 连续 KV Cache 的浪费

假设每个请求都预留最大长度 4096 token 的 KV Cache。

如果一个请求实际只用了 300 token，其余 3796 token 的 cache 空间就浪费了。

```text
reserved: 4096 tokens
used:      300 tokens
wasted:   3796 tokens
```

如果不预留最大长度，而是动态扩展连续空间，又会遇到碎片问题。

请求 A 释放一小段，请求 B 需要一大段，虽然总空闲显存够，但没有足够连续空间。

这和操作系统里内存分配遇到的问题很像。

## 17.3 操作系统分页的直觉

操作系统不会要求一个进程的虚拟内存全部连续映射到物理内存。

它会把内存切成 page：

```text
virtual page 0 -> physical page 7
virtual page 1 -> physical page 2
virtual page 2 -> physical page 9
```

进程看到的是连续虚拟地址，但物理内存可以不连续。

PagedAttention 借鉴了这个思想。

对一个请求来说，逻辑 token 序列是连续的：

```text
token 0, token 1, token 2, ..., token N
```

但它们对应的 KV Cache 物理 block 不需要连续。

## 17.4 逻辑块和物理块

PagedAttention 把一个请求的 KV Cache 按 token 维度切成固定大小的逻辑块。

假设 block size 是 16 token。

请求的 token 被分成：

```text
logical block 0: token 0-15
logical block 1: token 16-31
logical block 2: token 32-47
```

这些逻辑块映射到 GPU 上的物理 KV block：

```text
logical block 0 -> physical block 42
logical block 1 -> physical block 8
logical block 2 -> physical block 19
```

物理 block 可以不连续。

请求只需要维护一张 block table，记录逻辑块到物理块的映射。

## 17.5 Block Table

Block table 是理解 PagedAttention 的关键。

对一个请求：

```text
block_table = [42, 8, 19]
```

含义是：

```text
第 0 个逻辑 block 在物理 block 42
第 1 个逻辑 block 在物理 block 8
第 2 个逻辑 block 在物理 block 19
```

当 attention 需要读取历史 token 的 KV 时，它先根据 token 位置找到逻辑 block，再通过 block table 找到物理 block。

这一步类似地址翻译：

```text
token position -> logical block id -> physical block id -> KV memory address
```

用公式写，设 block size 为 `S`，请求 `i` 的 block table 为 `P_i`，token 位置为 `t`。

逻辑 block id：

```math
b_t=\left\lfloor\frac{t}{S}\right\rfloor
```

block 内 offset：

```math
o_t=t-Sb_t
```

物理 block id：

```math
p_t=P_i[b_t]
```

如果只看 token 维度，KV 读取位置可以抽象为：

```math
\mathrm{addr}(i,t)=(p_t,o_t)
```

请求 `i` 当前长度为 `T_i` 时，需要的 block 数为：

```math
B_i=\left\lceil\frac{T_i}{S}\right\rceil
```

最后一个 block 的块内浪费：

```math
R_i=S B_i-T_i
```

如果多个请求共享物理 block `p`，可以用引用计数 `c_p` 表示仍有多少请求引用它：

```math
c_p=\sum_i \mathbf{1}[p\in P_i]
```

只有当 `c_p=0` 时，物理 block 才能回到 free list。这个引用计数直觉是 prefix sharing 和 copy-on-write 的基础。

这就是 PagedAttention 的核心映射。

## 17.6 一个具体例子

假设 block size = 4。

请求 A 有 10 个 token：

```text
tokens: 0 1 2 3 | 4 5 6 7 | 8 9
```

它需要 3 个 logical blocks：

```text
logical block 0: token 0-3
logical block 1: token 4-7
logical block 2: token 8-11，其中 10-11 还没用
```

物理映射：

```text
block_table_A = [5, 11, 2]
```

请求 B 有 6 个 token：

```text
block_table_B = [7, 3]
```

请求 A 和 B 的物理块交错分布，但对每个请求来说，逻辑 token 序列仍然是连续的。

## 17.7 PagedAttention 如何减少浪费

PagedAttention 减少的是两类浪费。

第一，预留浪费。

不需要为每个请求一开始预留最大长度 cache。请求增长到新 block 时，再分配一个物理 block。

第二，碎片浪费。

物理 block 固定大小，可以从全局 free list 里复用。请求结束后，释放它占用的 blocks，其他请求可以继续使用。

相比连续分配，block 级分配更容易复用零散空间。

当然，PagedAttention 仍然有 block 内部浪费。

例如 block size 16，一个请求最后只用了 3 个 token 的 block，剩下 13 个 token 位置暂时浪费。但这通常比整段连续预留的浪费小得多。

## 17.8 Block Size 的 trade-off

block size 不是越小越好，也不是越大越好。

block size 小：

1. 内部浪费少。
2. block table 更长。
3. 地址映射和管理开销更大。
4. kernel 访问可能更碎。

block size 大：

1. block table 更短。
2. 管理开销小。
3. block 内部浪费更多。
4. 对短请求不友好。

所以 block size 是性能和显存利用率之间的 trade-off。

面试里如果只说“分块减少浪费”，还不够。要补一句：块大小会影响内部碎片、映射开销和 kernel 效率。

## 17.9 Attention 如何读取分页 KV

普通 attention 可能假设 KV Cache 是连续张量。

PagedAttention 下，历史 KV 分散在不同 physical blocks 中。

attention kernel 需要根据 block table 找到每个 token 对应的物理位置。

简化流程：

```text
for each query token:
  for each historical key/value token:
    logical_block = token_pos // block_size
    offset = token_pos % block_size
    physical_block = block_table[logical_block]
    read key/value from physical_block + offset
```

真实实现会用高效 CUDA kernel，而不是 Python 循环。

核心点是：PagedAttention 把地址映射逻辑融合进 attention 计算。

## 17.10 与 `past_key_values` 的区别

第 9 章的最小 KV Cache 使用 `past_key_values`。

它更像：

```text
每个 batch 持有一个连续 cache 张量
```

PagedAttention 更像：

```text
全局物理 KV block 池 + 每个请求的 block table
```

区别在于：

1. `past_key_values` 适合教学和简单 batch。
2. PagedAttention 适合动态请求、高并发和显存复用。
3. `past_key_values` 的 batch 顺序变化很麻烦。
4. PagedAttention 通过 block table 解耦请求逻辑顺序和物理存储。

这就是 vLLM 能支持 continuous batching 的基础之一。

## 17.11 与 continuous batching 的关系

PagedAttention 和 continuous batching 是互相支撑的。

continuous batching 要求：

1. 请求可以动态加入。
2. 请求可以动态退出。
3. cache 可以随请求增长。
4. finished 请求可以释放 cache。
5. batch row 和请求映射可以变化。

如果 KV Cache 必须是连续 batch 张量，这些操作会非常困难。

PagedAttention 通过 block table 和 block manager，让请求的 cache 独立管理，batch 只是每一轮执行时的临时组织方式。

这就是为什么 PagedAttention 不是孤立优化，而是 vLLM 调度系统的基础设施。

## 17.12 Prefix Sharing 的基础

PagedAttention 还为 prefix sharing 提供基础。

如果多个请求共享相同前缀，它们可以引用相同的物理 blocks。

例如：

```text
request A block_table: [1, 2, 3, 8]
request B block_table: [1, 2, 3, 9]
```

前 3 个 blocks 共享，后面分叉。

这类似 copy-on-write：共享部分不需要重复存储，只有分叉后的部分单独分配。

这对多轮对话、相同 system prompt、RAG 模板、批量评测都很有价值。

prefix caching 后面会单独讲，本章只需要知道：block-based cache 管理让共享更自然。

## 17.13 PagedAttention 不是免费午餐

PagedAttention 有明显收益，但也有代价。

代价包括：

1. attention kernel 更复杂。
2. 需要维护 block table。
3. 需要 block manager 分配和释放。
4. block size 选择影响性能。
5. 调试难度高于连续 cache。
6. 对硬件访问模式和 kernel 实现有要求。

工程上没有纯收益设计。vLLM 的价值在于这些复杂度换来了高并发场景下更好的显存利用和调度能力。

## 17.14 一个面试级类比

可以这样向面试官解释：

传统 KV Cache 像给每个请求分配一整段连续内存。请求长度不确定时，要么预留太多浪费，要么动态扩容导致碎片。

PagedAttention 像操作系统分页。请求看到的是连续逻辑 token 序列，但实际 KV Cache 存在固定大小的物理 blocks 中。通过 block table 做逻辑到物理的映射。这样请求可以动态增长，结束后释放 blocks，空闲 blocks 可以被其他请求复用。

这个类比清楚表达了 PagedAttention 的本质。

## 17.15 PagedAttention 地址翻译和可运行 demo

下面的 demo 不依赖外部库。它用 list 模拟一个 toy KV block pool：

1. 请求 A 写入 10 个 token，形成 3 个 logical blocks。
2. 查询 token 位置 `8`，通过 block table 翻译到 physical block 和 offset。
3. 请求 B 共享请求 A 的前 2 个 prefix blocks，再写入自己的分叉 token。
4. 释放请求 A 时，共享 blocks 的 ref count 保留，非共享 block 回到 free list。
5. 请求 C 复用请求 A 释放的 physical block。

```python
from dataclasses import dataclass, field


@dataclass
class RequestKV:
    request_id: str
    block_table: list[int] = field(default_factory=list)
    length: int = 0


class ToyPagedKVCache:
    def __init__(self, block_size, physical_blocks):
        self.block_size = block_size
        self.free_blocks = list(physical_blocks)
        self.storage = {block_id: [None] * block_size for block_id in physical_blocks}
        self.ref_count = {block_id: 0 for block_id in physical_blocks}
        self.trace = []

    def allocate_block(self):
        block_id = self.free_blocks.pop(0)
        self.storage[block_id] = [None] * self.block_size
        self.ref_count[block_id] = 1
        self.trace.append(("alloc", block_id))
        return block_id

    def retain_block(self, block_id):
        self.ref_count[block_id] += 1
        self.trace.append(("retain", block_id, self.ref_count[block_id]))

    def append_token(self, request, token):
        logical_block = request.length // self.block_size
        offset = request.length % self.block_size
        if offset == 0:
            request.block_table.append(self.allocate_block())
        physical_block = request.block_table[logical_block]
        self.storage[physical_block][offset] = token
        request.length += 1
        self.trace.append(
            ("write", request.request_id, token, logical_block, offset, physical_block)
        )

    def append_tokens(self, request, tokens):
        for token in tokens:
            self.append_token(request, token)

    def translate(self, request, token_position):
        logical_block = token_position // self.block_size
        offset = token_position % self.block_size
        physical_block = request.block_table[logical_block]
        value = self.storage[physical_block][offset]
        return {
            "token_position": token_position,
            "logical_block": logical_block,
            "offset": offset,
            "physical_block": physical_block,
            "value": value,
        }

    def fork_shared_prefix(self, source, new_request_id, prefix_blocks):
        child = RequestKV(new_request_id)
        for physical_block in source.block_table[:prefix_blocks]:
            self.retain_block(physical_block)
            child.block_table.append(physical_block)
        child.length = prefix_blocks * self.block_size
        return child

    def release_request(self, request):
        released = []
        for physical_block in request.block_table:
            self.ref_count[physical_block] -= 1
            if self.ref_count[physical_block] == 0:
                self.storage[physical_block] = [None] * self.block_size
                self.free_blocks.insert(0, physical_block)
                released.append(physical_block)
        self.trace.append(("release", request.request_id, released))
        request.block_table = []
        request.length = 0
        return released

    def internal_waste(self, requests):
        used_by_block = {}
        for request in requests:
            for block_index, physical_block in enumerate(request.block_table):
                start = block_index * self.block_size
                used = max(0, min(self.block_size, request.length - start))
                used_by_block[physical_block] = max(used_by_block.get(physical_block, 0), used)
        allocated_slots = len(used_by_block) * self.block_size
        used_slots = sum(used_by_block.values())
        return {
            "allocated_slots": allocated_slots,
            "used_slots": used_slots,
            "wasted_slots": allocated_slots - used_slots,
        }


def run_paged_attention_demo():
    cache = ToyPagedKVCache(block_size=4, physical_blocks=[7, 3, 11, 5, 9, 2, 13])
    req_a = RequestKV("A")
    cache.append_tokens(req_a, list("abcdefghij"))
    translations = [cache.translate(req_a, pos) for pos in [0, 4, 8, 9]]

    req_b = cache.fork_shared_prefix(req_a, "B", prefix_blocks=2)
    cache.append_tokens(req_b, list("XY"))
    shared_ref_counts = {block: cache.ref_count[block] for block in req_a.block_table[:2]}

    released_a = cache.release_request(req_a)
    ref_counts_after_a = {block: cache.ref_count[block] for block in [7, 3, 11]}
    req_c = RequestKV("C")
    cache.append_tokens(req_c, list("mnop"))

    waste = cache.internal_waste([req_b, req_c])
    summary = {
        "req_b_table": req_b.block_table,
        "req_c_table": req_c.block_table,
        "translations": translations,
        "shared_ref_counts": shared_ref_counts,
        "released_a": released_a,
        "ref_counts_after_a": ref_counts_after_a,
        "free_head": cache.free_blocks[:4],
        "waste": waste,
    }
    gates = {
        "address_translation": translations[2] == {
            "token_position": 8,
            "logical_block": 2,
            "offset": 0,
            "physical_block": 11,
            "value": "i",
        },
        "prefix_blocks_shared": req_b.block_table[:2] == [7, 3]
        and shared_ref_counts == {7: 2, 3: 2},
        "partial_release_keeps_shared": ref_counts_after_a == {7: 1, 3: 1, 11: 0},
        "freed_block_reused": req_c.block_table == [11],
        "internal_waste_visible": waste == {
            "allocated_slots": 16,
            "used_slots": 14,
            "wasted_slots": 2,
        },
    }
    gates["paged_attention_gate"] = all(gates.values())
    return summary, cache.trace[-10:], gates


summary, trace_tail, gates = run_paged_attention_demo()
print("paged_kv_summary=", summary)
print("trace_tail=", trace_tail)
print("paged_attention_gates=", gates)
```

一组稳定输出：

```text
paged_kv_summary= {'req_b_table': [7, 3, 5], 'req_c_table': [11], 'translations': [{'token_position': 0, 'logical_block': 0, 'offset': 0, 'physical_block': 7, 'value': 'a'}, {'token_position': 4, 'logical_block': 1, 'offset': 0, 'physical_block': 3, 'value': 'e'}, {'token_position': 8, 'logical_block': 2, 'offset': 0, 'physical_block': 11, 'value': 'i'}, {'token_position': 9, 'logical_block': 2, 'offset': 1, 'physical_block': 11, 'value': 'j'}], 'shared_ref_counts': {7: 2, 3: 2}, 'released_a': [11], 'ref_counts_after_a': {7: 1, 3: 1, 11: 0}, 'free_head': [9, 2, 13], 'waste': {'allocated_slots': 16, 'used_slots': 14, 'wasted_slots': 2}}
trace_tail= [('retain', 3, 2), ('alloc', 5), ('write', 'B', 'X', 2, 0, 5), ('write', 'B', 'Y', 2, 1, 5), ('release', 'A', [11]), ('alloc', 11), ('write', 'C', 'm', 0, 0, 11), ('write', 'C', 'n', 0, 1, 11), ('write', 'C', 'o', 0, 2, 11), ('write', 'C', 'p', 0, 3, 11)]
paged_attention_gates= {'address_translation': True, 'prefix_blocks_shared': True, 'partial_release_keeps_shared': True, 'freed_block_reused': True, 'internal_waste_visible': True, 'paged_attention_gate': True}
```

这个 demo 的关键证据：

1. token position `8` 被翻译成 logical block `2`、offset `0`、physical block `11`。
2. 请求 B 的前两个 physical blocks 与请求 A 共享，`ref_count` 从 1 变成 2。
3. 释放请求 A 后，shared blocks `7` 和 `3` 仍保留，非共享 block `11` 回到 free list。
4. 请求 C 随后复用了 physical block `11`。
5. 当前活跃请求只浪费 `2` 个 block slot，说明分页不能消灭浪费，但能把浪费压缩到尾块级别。

## 17.16 常见误区

误区一：PagedAttention 只是更快的 attention。

它更重要的是 KV Cache 的分页式内存管理，kernel 只是实现这一管理方式所需的执行机制。

误区二：PagedAttention 完全没有浪费。

仍然有 block 内部碎片，只是比大段连续预留浪费少。

误区三：block size 越小越好。

block 小会减少内部浪费，但增加 block table、调度和 kernel 访问开销。

误区四：有了 PagedAttention 就不需要 scheduler。

PagedAttention 管 cache 存储，scheduler 管每轮执行谁。两者互相配合。

误区五：PagedAttention 和 prefix caching 没关系。

block-based cache 管理让共享 prefix 的物理 block 复用更自然。

## 17.17 面试追问

1. PagedAttention 的核心思想是什么？
2. 为什么连续 KV Cache 管理会浪费显存？
3. logical block、physical block、block table 分别是什么？
4. PagedAttention 如何借鉴操作系统分页？
5. block size 有什么 trade-off？
6. PagedAttention 和 continuous batching 有什么关系？
7. PagedAttention 是否完全消除了显存浪费？
8. 为什么 prefix sharing 更适合建立在 block-based cache 上？

参考回答思路：

1. 先说 PagedAttention 是 KV Cache 的分页式管理。
2. 再说请求逻辑 token 连续，但物理 KV blocks 可以不连续，通过 block table 映射。
3. 然后解释收益：减少预留浪费和碎片，支持动态增长、释放和共享。
4. 最后补 trade-off：block size、kernel 复杂度、block manager 开销。

## 17.18 小练习

1. 假设 block size 为 16，一个 100 token 请求需要几个 block？最后一个 block 浪费多少 token 位置？
2. 画出两个请求共享前三个 prefix blocks 后分别分叉的 block table。
3. 解释为什么请求结束后释放 blocks 比释放一段连续 cache 更容易复用。
4. 思考 block size 从 16 改成 8，会影响哪些指标？
5. 用一句话解释 PagedAttention 和普通 `past_key_values` 的区别。

## 17.19 本章小结

本章讲清了 PagedAttention 的核心思想。

PagedAttention 借鉴操作系统分页，把请求的逻辑 KV Cache 切成固定大小的 logical blocks，并映射到 GPU 上的 physical KV blocks。Block table 负责逻辑到物理的映射。这样可以减少连续分配带来的预留浪费和碎片，支持请求动态增长、完成释放和 prefix sharing。

下一章我们会继续深入 KV Cache Block Manager，讲它如何分配、释放、复用和追踪这些 physical blocks。
