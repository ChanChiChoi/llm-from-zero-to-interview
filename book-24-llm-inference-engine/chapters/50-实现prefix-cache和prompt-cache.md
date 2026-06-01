# 第 50 章 实现 prefix cache 和 prompt cache

上一章我们把 mini engine 的 KV cache 从 request 私有 list 升级成了 paged KV cache。

有了 block pool、block table、slot mapping 和 BlockManager 之后，下一步自然就是复用已经算过的 prefix。

这就是本章要做的事情：在 mini engine 中实现第一版 prefix cache，并讲清它和 prompt cache 的关系。

第 23 章已经系统讲过 Prefix Caching 与 Prompt Cache 的概念，第 28 章讲过 SGLang RadixAttention 的 prefix sharing。

本章不再重复原理，而是回答一个工程问题：

```text
如果你已经有 paged KV cache，怎样加一个能工作的 hash-based prefix cache？
```

## 50.1 本章目标

读完本章，你应该能讲清：

1. prompt cache 和 runtime prefix cache 的边界。
2. 为什么第一版只缓存 full KV blocks。
3. block hash 需要包含哪些信息。
4. 新请求如何查找最长可复用 prefix。
5. cache hit 后 request 的 block table 如何初始化。
6. 命中 prefix 后为什么只 prefill suffix。
7. cached block 的 ref count 如何维护。
8. 请求结束后哪些 blocks 可以进入 cache，哪些必须释放。
9. prefix cache 如何影响 scheduler、TTFT 和显存。
10. 如何设计日志、指标和压测验证 prefix cache 生效。

面试里最容易出现的错误是：把 prompt cache 说成一个字典 `cache[prompt] = output`。

这不是本章要实现的东西。

我们要实现的是 runtime 内部的 KV block 复用。

## 50.2 先划清边界：prompt cache vs prefix cache

Prompt cache 是更宽泛的产品或平台层概念。

它关心的是：

```text
这个 prompt 的某个前缀是否可以复用？
```

它可能缓存：

1. tokenized prompt。
2. 文档解析结果。
3. RAG 检索结果。
4. embedding。
5. runtime 的 KV blocks。
6. 完整响应。

Prefix cache 是 runtime 内部更具体的机制。

它关心的是：

```text
这段 token prefix 对应的 KV blocks 是否已经在 block pool 中？
```

本章实现的是第二种：

```text
hash-based KV block prefix cache
```

也就是：

```text
tokens -> full block hashes -> cached physical KV blocks -> request block table prefix
```

## 50.3 为什么必须建立在 paged KV cache 上

如果 KV cache 还是每个 request 自己的 list，prefix cache 会很别扭。

因为复用一段 prefix 意味着：

```text
多个 request 同时引用同一段已经计算好的 KV。
```

这需要三个能力：

1. KV 被切成可引用的 block。
2. request 可以通过 block table 引用这些 block。
3. block 有 ref count，知道什么时候能释放。

这正是第 49 章 paged KV cache 提供的基础。

没有 paged KV cache，也可以做粗糙的 prompt cache，但很难做细粒度、在线、可调度的 KV prefix cache。

## 50.4 第一版范围

第一版不要做太多。

我们只实现：

1. 只缓存 full blocks。
2. 只支持从 prompt 开头开始的 prefix hit。
3. 只做 hash exact match。
4. 不做 radix tree。
5. 不做 CPU offload。
6. 不做复杂 eviction。
7. 不做 partial block cache。
8. 不做跨模型、跨 tokenizer、跨租户共享。

目标是先跑通这条路径：

```text
request A prefill 完成 full blocks -> blocks 加入 prefix cache
request B 发现相同前缀 -> 复用 A 的 cached blocks -> 只 prefill 未命中的 suffix
```

这已经足够解释 vLLM-like prefix caching 的主干。

## 50.5 最小数据结构

在第 49 章的基础上，为 block 对象增加元数据。

```python
class KVCacheBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.block_hash = None
        self.is_cached = False
```

BlockManager 维护所有 blocks 和 cache map。

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_blocks = list(range(num_blocks))
        self.cached_blocks = {}
```

其中：

1. `blocks`：物理 block 对象池。
2. `free_blocks`：当前完全空闲、可分配的 block id。
3. `cached_blocks`：`block_hash -> block_id`。
4. `ref_count`：当前有多少 request 或 cache entry 引用这个 block。
5. `is_cached`：这个 block 是否保留在 prefix cache 中。

注意这里第一版为了教学，把 cache entry 和 block 绑定得很简单。

真实系统里还会有 free queue、evictable queue、LRU 时间戳、hash collision 处理和多级缓存策略。

## 50.6 ref count 的含义

加入 prefix cache 后，ref count 不再只是“有多少 running request 引用”。

它还要考虑 cache 本身的引用。

一种简单定义是：

```text
ref_count = active_request_refs + cached_ref
```

例如：

```text
block 7 被 request A 使用，还在 prefix cache 中
active_request_refs = 1
cached_ref = 1
ref_count = 2
```

request A 结束后：

```text
active_request_refs = 0
cached_ref = 1
ref_count = 1
```

block 不能放回 free list，因为它还作为 cache 保存。

只有 cache entry 被淘汰后：

```text
cached_ref = 0
ref_count = 0
```

block 才能真正进入 free list。

教学版也可以把 active ref 和 cached ref 拆成两个字段：

```python
class KVCacheBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.active_refs = 0
        self.cached = False
        self.block_hash = None
```

这样更容易 debug。

但面试中通常讲 ref count 就够了。

## 50.7 block hash 包含什么

不能只 hash 当前 block tokens。

最小安全 hash 至少包含：

1. parent hash。
2. block token ids。
3. model id。
4. tokenizer id 或 tokenizer version。
5. cache salt 或 tenant id。
6. LoRA adapter id。
7. 多模态输入 hash。

教学版可以先实现前三个：

```python
def hash_block(parent_hash, block_tokens, model_id):
    return hash((parent_hash, tuple(block_tokens), model_id))
```

更完整一点：

```python
def hash_block(parent_hash, block_tokens, extra_hashes):
    return hash((parent_hash, tuple(block_tokens), tuple(extra_hashes)))
```

其中 `extra_hashes` 可以包含：

```python
extra_hashes = (
    model_id,
    tokenizer_id,
    lora_id,
    tenant_salt,
)
```

为什么要 parent hash？

因为 causal attention 下，同一个 block tokens 出现在不同前缀之后，KV 语义不同。

```text
A B C D + X Y Z
P Q R S + X Y Z
```

最后 `[X Y Z]` 一样，不代表它们的 KV 可以共享。

parent hash 把前文链路编码进当前 block hash。

## 50.8 只缓存 full blocks

第一版只缓存 full blocks。

假设 block size 为 4：

```text
prompt: A B C D E F
blocks:
  block 0: A B C D  full
  block 1: E F _ _  partial
```

只把 block 0 放入 prefix cache。

block 1 不缓存。

原因是 partial block 会继续变化。

如果后续追加 G H，block 1 从：

```text
E F _ _
```

变成：

```text
E F G H
```

它的 hash 和内容都变了。

只缓存 full blocks 可以让 cache entry 稳定、复用边界清晰、实现简单。

代价是：共享前缀最后不足一个 block 的部分不能命中，需要重新 prefill。

## 50.9 把 prompt 切成 full blocks

查 cache 前，需要把 prompt tokens 切成 full blocks。

```python
def iter_full_token_blocks(token_ids: list[int], block_size: int):
    n_full_blocks = len(token_ids) // block_size
    for i in range(n_full_blocks):
        start = i * block_size
        end = start + block_size
        yield token_ids[start:end]
```

如果 prompt 长度为 10，block size 为 4：

```text
full blocks:
  tokens 0-3
  tokens 4-7

partial suffix:
  tokens 8-9
```

prefix cache 最多命中前 8 个 tokens。

剩下 2 个 tokens 仍然要 prefill。

## 50.10 查找最长命中 prefix

新请求进入后，BlockManager 尝试从开头逐 block 命中。

```python
def find_cached_prefix(self, token_ids, extra_hashes):
    parent_hash = None
    hit_block_ids = []
    hit_hashes = []

    for block_tokens in iter_full_token_blocks(token_ids, self.block_size):
        block_hash = hash_block(parent_hash, block_tokens, extra_hashes)
        block_id = self.cached_blocks.get(block_hash)

        if block_id is None:
            break

        hit_block_ids.append(block_id)
        hit_hashes.append(block_hash)
        parent_hash = block_hash

    hit_tokens = len(hit_block_ids) * self.block_size
    return hit_block_ids, hit_hashes, hit_tokens, parent_hash
```

注意它必须从第一个 block 开始连续命中。

不能跳过中间 miss。

例如：

```text
block 0 hit
block 1 miss
block 2 即使 hash 存在，也不能复用
```

因为 block 2 的 parent chain 已经断了。

## 50.11 cache hit 后如何初始化 request

假设新请求 prompt 长度为 1000，block size 为 16，命中前 768 tokens。

也就是命中 48 个 full blocks。

初始化 request：

```python
request.block_table = hit_block_ids.copy()
request.num_computed_tokens = hit_tokens
request.prefix_cache_hit_tokens = hit_tokens
request.prefix_cache_parent_hash = parent_hash
```

然后增加这些 blocks 的 active ref：

```python
for block_id in hit_block_ids:
    self.blocks[block_id].ref_count += 1
```

这样 request 后续 attention 就能通过 block table 读到缓存的历史 KV。

它不需要重新计算前 768 个 tokens。

## 50.12 只 prefill suffix

命中 prefix 后，scheduler 不能再从 position 0 开始 prefill。

它应该从：

```python
start_pos = request.num_computed_tokens
```

开始。

例如：

```text
prompt length = 1000
cache hit = 768

需要 prefill:
  positions 768-999
```

如果 chunked prefill 每轮最多 128 tokens，则第一轮 suffix prefill：

```text
positions 768-895
```

第二轮：

```text
positions 896-999
```

这要求 BatchBuilder 支持非零起点的 prefill。

上一章的 `ScheduleItem.start_pos` 正是为这个准备的。

## 50.13 prefill suffix 前还要分配 blocks

命中 prefix 后，request.block_table 已经包含 cached blocks。

但 suffix 仍然需要新 blocks。

例如 block size 为 16：

```text
hit tokens = 768
next suffix token position = 768
```

如果 768 正好是 block 边界，那么 suffix 第一个 token 需要新 block。

调度前仍然要调用：

```python
block_manager.allocate_until(request, target_tokens)
```

其中 target tokens 是本轮 suffix chunk 的结束位置。

```python
target_tokens = start_pos + chunk_tokens
```

如果 cache hit tokens 不是完整 prompt，后续路径和普通 chunked prefill 完全一样。

## 50.14 什么时候把 block 加入 cache

一个 block 能进入 prefix cache，至少要满足：

1. 它是 full block。
2. 它的 KV 已经计算完成。
3. 它的 block hash 已经确定。
4. 它没有被写坏或复用给其他未完成语义。

最简单时机：prefill chunk 完成后，把其中新完成的 full blocks 注册进 cache。

例如 block size 为 4，本次 prefill 完成后 computed tokens 从 6 变成 10：

```text
之前 full blocks: floor(6 / 4) = 1
现在 full blocks: floor(10 / 4) = 2
新增可缓存 block: logical block 1
```

代码：

```python
def cache_new_full_blocks(self, request, old_computed, new_computed, extra_hashes):
    old_full = old_computed // self.block_size
    new_full = new_computed // self.block_size

    parent_hash = request.prefix_cache_parent_hash
    if parent_hash is None and old_full > 0:
        parent_hash = request.block_hashes[old_full - 1]

    for logical_block_id in range(old_full, new_full):
        start = logical_block_id * self.block_size
        end = start + self.block_size
        block_tokens = request.prompt_token_ids[start:end]
        block_hash = hash_block(parent_hash, block_tokens, extra_hashes)
        physical_block_id = request.block_table[logical_block_id]

        self.register_cached_block(physical_block_id, block_hash)
        request.block_hashes.append(block_hash)
        parent_hash = block_hash
```

这段代码是教学版，真实实现要处理 output tokens、分支、hash collision 和已有 cache entry。

第一版可以只缓存 prompt prefill 的 full blocks，不缓存 decode 生成的 blocks。

## 50.15 register_cached_block

注册 cache entry 时，需要处理两个问题：

1. 这个 hash 是否已经存在。
2. 这个 block 是否已经被缓存。

最小实现：

```python
def register_cached_block(self, block_id: int, block_hash):
    block = self.blocks[block_id]

    if block_hash in self.cached_blocks:
        return

    block.block_hash = block_hash
    block.is_cached = True
    block.ref_count += 1
    self.cached_blocks[block_hash] = block_id
```

这里 `ref_count += 1` 表示 cache 自己持有一份引用。

如果 request 后续结束，它释放 active ref，但 cached ref 还在，block 不会回到 free list。

如果发现 hash 已经存在，第一版可以直接不注册。

更严谨的实现要验证 block tokens 和 extra hashes，防止 hash collision。

## 50.16 request 结束时如何释放

没有 prefix cache 时，request 结束后所有 blocks 都释放回 free list。

有 prefix cache 后，不一定。

释放 active ref：

```python
def free_request(self, request):
    for block_id in request.block_table:
        block = self.blocks[block_id]
        block.ref_count -= 1

        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    request.block_table = []
```

如果某个 block 在 cache 中，`ref_count` 至少还有 cached ref，因此不会进入 free list。

这就是 prefix cache 牺牲一部分显存换 TTFT 的地方。

## 50.17 cache eviction 第一版怎么做

如果 cached blocks 永远不淘汰，显存迟早被 cache 占满。

第一版可以做一个非常简单的策略：当 free blocks 低于水位线时，清空一部分 cached blocks。

```python
def evict_one_cached_block(self):
    for block_hash, block_id in list(self.cached_blocks.items()):
        block = self.blocks[block_id]

        if block.ref_count == 1 and block.is_cached:
            del self.cached_blocks[block_hash]
            block.is_cached = False
            block.block_hash = None
            block.ref_count -= 1
            self.free_blocks.append(block_id)
            return True

    return False
```

这里 `ref_count == 1` 表示只有 cache 自己引用，没有 active request 正在用。

不能淘汰 active request 正在使用的 block。

水位线策略：

```python
def maybe_evict_cache(self, min_free_blocks):
    while len(self.free_blocks) < min_free_blocks:
        if not self.evict_one_cached_block():
            break
```

这不是高性能策略，但教学足够。

后续可以升级成 LRU、按 prefix 长度、按命中次数、按租户 quota 淘汰。

## 50.18 prefix cache 对 scheduler 的影响

prefix cache 会改变 scheduler 看到的请求长度。

原本一个请求 prompt 长度是：

```text
prompt_len = 8000
```

命中 7000 tokens 后，真实需要 prefill 的 suffix 是：

```text
remaining_prefill = 1000
```

所以 scheduler 的 token budget 应该消耗 suffix tokens，而不是完整 prompt tokens。

```python
remaining = len(request.prompt_token_ids) - request.num_computed_tokens
chunk_tokens = min(remaining, max_prefill_tokens_per_step, token_budget)
```

这会直接改善 TTFT。

因为长 prompt 的大部分 prefill 被跳过，请求更快进入 decode。

但它也可能增加显存压力。

因为 cached blocks 被保留，free blocks 下降。

所以 prefix cache 不是纯收益，它是计算换显存。

## 50.19 cache hit 后的状态机

一个命中 prefix cache 的 request 状态流如下：

```text
WAITING
  -> lookup_prefix_cache
  -> attach cached blocks
  -> num_computed_tokens = hit_tokens
  -> PREFILLING_SUFFIX
  -> DECODING
  -> FINISHED
  -> release active refs
```

没有命中时：

```text
WAITING
  -> num_computed_tokens = 0
  -> PREFILLING_FULL_PROMPT
  -> DECODING
  -> FINISHED
```

因此 prefix cache 不应该引入一套完全不同的调度路径。

它只是改变了 request 的初始 `num_computed_tokens` 和 `block_table`。

后续 suffix prefill、decode、finish 都复用原来的流程。

这是实现上最重要的简化。

## 50.20 BatchBuilder 需要注意什么

cache hit 后，BatchBuilder 会收到一个从非零 start position 开始的 prefill item。

例如：

```text
request A:
  prompt_len = 1000
  num_computed_tokens = 768
  schedule prefill positions 768-895
```

BatchBuilder 必须：

1. input_ids 取 positions 768-895 对应的 tokens。
2. positions 填真实逻辑位置 768-895。
3. slot_mapping 根据 block table 计算物理写入位置。
4. block_table 包含 cached prefix blocks 和 suffix 新 blocks。
5. seq_len 至少覆盖当前 chunk 可见的上下文。

最常见 bug 是把 positions 从 0 开始重置。

这会导致 RoPE position、attention mask 和 KV 写入全部错。

## 50.21 OutputProcessor 需要注意什么

prefill suffix 完成后，OutputProcessor 更新：

```python
old_computed = request.num_computed_tokens
new_computed = old_computed + item.num_tokens
request.num_computed_tokens = new_computed
```

然后注册新增 full blocks：

```python
block_manager.cache_new_full_blocks(
    request,
    old_computed,
    new_computed,
    extra_hashes,
)
```

如果 `new_computed == len(prompt_token_ids)`，说明 prompt prefill 完成，可以采样 first token 或进入 decode。

prefix cache hit 的 request 不应该跳过 suffix 的 logits 处理。

如果 suffix 长度为 0，也就是完整 prompt 都命中 full blocks，这时仍然要考虑如何得到 first token logits。

第一版可以保守处理：至少执行最后一个 token 的 lightweight prefill 或 recompute 一个很短 suffix。

真实系统会更精细地处理全命中场景。

## 50.22 全 prompt 命中怎么办

假设 prompt 长度正好是 block size 的整数倍，并且所有 full blocks 都命中。

此时：

```text
remaining_prefill = 0
```

但要生成第一个 output token，需要 prompt 最后一个位置的 logits。

KV cache 里保存的是 K/V，不一定保存 logits。

所以仅仅命中 KV blocks，不代表可以不做任何 forward 直接采样。

教学版有两个选择。

第一，保守方案：重新跑最后一个 prompt token，得到 logits。

```text
prefill positions: prompt_len - 1
```

这会多算一点，但实现简单。

第二，扩展 cache：同时缓存最后 token logits。

这更接近 prompt result cache，但复杂度更高。

第一版建议采用保守方案。

面试里要讲清楚：

```text
Prefix cache 复用 KV，不等于一定缓存了 logits 或最终输出。
```

## 50.23 和 result cache 不一样

Prefix cache 不是 result cache。

Result cache 是：

```text
同一个完整请求 -> 直接返回之前生成的完整回答
```

Prefix cache 是：

```text
共享 prompt 前缀 -> 复用前缀 KV -> 后续仍然要继续 prefill suffix 和 decode
```

区别：

| 类型 | 复用内容 | 是否还要生成 | 风险 |
|---|---|---|---|
| prefix cache | KV blocks | 要 | 显存占用、错误复用 |
| result cache | 完整输出 | 不一定 | 过期、个性化、随机采样不一致 |

如果用户设置 temperature > 0，同一个 prompt 也可能希望得到不同答案。

这时 result cache 直接返回旧答案可能不符合预期，但 prefix cache 仍然可以安全复用 deterministic 的 prompt KV。

## 50.24 cache salt 和租户隔离

多租户系统不能随便跨用户共享 prefix cache。

即使两个租户 prompt 完全一样，如果共享 cache 导致时延差异可观察，也可能泄露“另一个租户是否请求过相同内容”。

解决方法之一是把 `tenant_salt` 放进 hash。

```python
extra_hashes = (
    model_id,
    tokenizer_id,
    tenant_salt,
)
```

这样不同租户即使 token ids 一样，也不会命中同一个 cache entry。

如果业务允许跨租户共享，可以用更粗粒度的 salt。

关键是：这不是技术默认选择，而是安全和产品策略。

## 50.25 LoRA 和模型版本

LoRA adapter 会改变模型权重。

同样 token ids，在不同 LoRA 下计算出的 KV 不同。

所以 LoRA id 必须进入 hash。

```python
extra_hashes = (
    model_id,
    tokenizer_id,
    lora_id,
    tenant_salt,
)
```

模型版本也一样。

如果模型从 `model-v1` 热更新到 `model-v2`，旧 KV cache 不能继续复用。

最简单做法：model id 或 weights version 进入 hash。

模型切换时也可以直接清空 prefix cache。

## 50.26 多模态输入

多模态模型里，prompt token 序列可能包含图片占位符。

例如：

```text
<image> 请描述这张图
```

两个请求 token ids 可能一样，但图片内容不同。

如果只 hash token ids，就会错误复用 KV。

因此 extra hashes 要包含媒体内容 hash 或视觉 embedding 版本 hash。

```python
extra_hashes = (
    model_id,
    tokenizer_id,
    image_hash,
    tenant_salt,
)
```

第一版 mini engine 如果只支持文本，可以暂时不实现。

但面试回答时要主动提到这个坑。

## 50.27 日志设计

prefix cache 必须有命中日志。

request 级别：

```text
request=req-10 event=prefix_cache_lookup prompt_tokens=8192 hit_tokens=6144 hit_blocks=384 hit_ratio=0.75
request=req-10 event=attach_cached_blocks blocks=384 computed_tokens=6144
request=req-10 event=prefill_suffix start=6144 tokens=512
```

block 级别：

```text
block=77 event=cache_register hash=abc ref_count=2
block=77 event=cache_hit hash=abc ref_count=3
block=77 event=cache_evict hash=abc ref_count=0
```

scheduler 级别：

```text
step=201 prefix_hit_tokens=24576 prefill_suffix_tokens=4096 cache_hit_reqs=7 cache_miss_reqs=3 free_blocks=1024 cached_blocks=8192
```

这些日志能回答：

1. cache 到底有没有命中？
2. 命中了多少 tokens？
3. TTFT 改善是否来自 prefix cache？
4. cached blocks 是否挤压正常请求？
5. eviction 是否太频繁？

## 50.28 指标设计

核心指标：

| 指标 | 含义 |
|---|---|
| prefix_cache_hit_requests | 命中至少一个 block 的请求数 |
| prefix_cache_miss_requests | 完全未命中的请求数 |
| prefix_cache_hit_tokens | 被复用的 token 数 |
| prefix_cache_total_prompt_tokens | 总 prompt token 数 |
| prefix_cache_hit_ratio_tokens | hit tokens / prompt tokens |
| cached_blocks | 当前缓存 blocks 数 |
| cache_evictions | cache block 淘汰次数 |
| cache_ref_count_active | 被 active request 引用的 cached blocks |
| ttft_cache_hit_p50/p99 | 命中请求的 TTFT |
| ttft_cache_miss_p50/p99 | 未命中请求的 TTFT |

不要只看 request hit ratio。

一个请求命中 1 个 block 和命中 1000 个 blocks，价值完全不同。

更重要的是 token hit ratio：

```text
prefix_cache_hit_tokens / prefix_cache_total_prompt_tokens
```

## 50.29 压测场景一：固定 system prompt

workload：

```text
system prompt: 2K tokens 固定
user query: 32-128 tokens 随机
output: 128 tokens
```

预期：

1. 第一批请求 miss。
2. 后续请求命中 system prompt 对应 full blocks。
3. TTFT 明显下降。
4. prefill tokens/s 中真实计算 tokens 下降。
5. cached blocks 保持稳定。

这是最适合验证第一版 prefix cache 的场景。

## 50.30 压测场景二：长文档多问题

workload：

```text
shared document: 8K tokens
question: 64 tokens
output: 256 tokens
```

请求格式：

```text
[system][document][question A]
[system][document][question B]
[system][document][question C]
```

预期：

1. shared document 前缀命中率高。
2. TTFT cache hit 请求显著低于 miss 请求。
3. token budget 更多留给未命中 suffix。
4. free blocks 会下降，因为 cached blocks 被保留。

这个场景能体现 prefix cache 的主要价值。

## 50.31 压测场景三：低复用随机 prompt

workload：

```text
prompt 完全随机
共享前缀很少
```

预期：

1. cache hit ratio 很低。
2. TTFT 几乎不改善。
3. cached blocks 可能占用显存。
4. eviction 可能频繁发生。

这个场景用来验证 prefix cache 的反面：

```text
没有复用时，cache 不应该明显拖垮系统。
```

如果随机 prompt 下性能明显变差，说明 cache 注册、hash、eviction 或 ref count 开销过大。

## 50.32 常见 bug

bug 一：block hash 没有包含 parent hash。

```text
结果：相同 block tokens 在不同前缀下被错误复用。
```

bug 二：cache hit 后没有增加 ref count。

```text
结果：cached block 可能被释放或分配给其他请求，导致 KV 被覆盖。
```

bug 三：request 结束时把 cached block 直接放回 free list。

```text
结果：cache map 还指向这个 block，但 block 已经被新请求复用。
```

bug 四：命中 prefix 后 prefill 仍从 0 开始。

```text
结果：重复计算 prefix，甚至覆盖 cached blocks。
```

bug 五：命中 prefix 后 positions 从 0 重新编号。

```text
结果：RoPE position 和 attention mask 错误。
```

bug 六：只统计 request hit ratio，不统计 token hit ratio。

```text
结果：看起来命中率很高，但实际节省 token 很少。
```

bug 七：没有区分 model、tokenizer、LoRA 或 tenant salt。

```text
结果：跨模型或跨租户错误复用 KV。
```

bug 八：全 prompt 命中时直接 decode，但没有可用 logits。

```text
结果：无法采样 first token，或者错误使用旧 logits。
```

## 50.33 面试高频问题

问题一：prefix cache 主要优化 TTFT 还是 TPOT？

回答要点：主要优化 TTFT，因为它减少重复前缀的 prefill 计算，让请求更快进入 first token。TPOT 发生在 decode 阶段，prefix cache 对它的影响通常是间接的。

问题二：为什么 vLLM-like prefix cache 通常只缓存 full blocks？

回答要点：full block 边界稳定，hash 和复用简单；partial block 会继续变化，缓存和一致性更复杂。代价是共享前缀最后不足一个 block 的部分不能命中。

问题三：block hash 为什么要包含 parent hash？

回答要点：因为 causal attention 下 KV 依赖完整前文。同一个 block tokens 出现在不同 prefix 后面，语义不同，不能只 hash当前 block tokens。parent hash 把整条 prefix chain 编码进来。

问题四：prefix cache 命中后 scheduler 怎么变？

回答要点：request 的 `num_computed_tokens` 初始化为 hit tokens，block table 前缀直接引用 cached blocks。scheduler 只需要调度未命中的 suffix prefill，token budget 消耗 suffix tokens，而不是完整 prompt tokens。

问题五：prefix cache 为什么会增加显存压力？

回答要点：cached blocks 即使没有 active request 使用，也会保留在 block pool 中等待复用。它减少计算但占用 KV blocks，因此需要 eviction、水位线和指标监控。

问题六：prefix cache 和 result cache 有什么区别？

回答要点：prefix cache 复用 prompt 前缀的 KV blocks，后续仍然要 prefill suffix 和 decode；result cache 复用完整输出，可能直接返回答案。prefix cache 通常更适合 stochastic generation，因为它不固定最终输出。

## 50.34 标准回答模板

如果面试官问“你如何在 paged KV cache 上实现 prefix cache”，可以这样回答：

```text
我会先基于 paged KV cache 做 hash-based prefix cache。KV cache 已经被切成 fixed-size blocks，每个 request 有自己的 block table，因此 prefix cache 可以缓存已经计算完成的 full blocks，并让新请求的 block table 前缀直接引用这些 cached physical blocks。

具体数据结构上，BlockManager 维护 cached_blocks: block_hash -> block_id，每个 block 维护 ref count、block hash 和 cached 标记。block hash 不能只包含当前 block tokens，还要包含 parent hash 和 extra hashes，比如 model id、tokenizer version、LoRA id、tenant salt、多模态输入 hash等，避免错误复用。

新请求到来后，我会按 block size 从 prompt 开头切 full blocks，逐块计算链式 hash，查找最长连续命中的 cached blocks。命中后，把这些 block ids 放进 request.block_table，把 request.num_computed_tokens 设置为 hit tokens，并增加这些 blocks 的 ref count。之后 scheduler 只调度未命中的 suffix prefill，BatchBuilder 从真实 start position 构造 input ids、positions、slot mapping 和 block table。

prefill suffix 完成后，OutputProcessor 会把新增完成的 full blocks 注册进 prefix cache。request 结束时只释放 active refs，cached blocks 如果还有 cache ref 不能回到 free list。显存紧张时再通过 eviction 淘汰 ref_count 只有 cache 自己持有的 blocks。

验证时我会看 prefix_cache_hit_tokens、token hit ratio、TTFT hit/miss 对比、cached blocks、eviction 次数和 free blocks。固定 system prompt、长文档多问题这类 workload 应该显著降低 TTFT；随机 prompt workload 下 hit ratio 低，cache 不应该明显拖慢系统。
```

## 50.35 小练习

1. 给 `KVCacheBlock` 增加 `ref_count`、`block_hash`、`is_cached` 字段。
2. 给 `BlockManager` 增加 `cached_blocks` 字典。
3. 实现 `iter_full_token_blocks(token_ids, block_size)`。
4. 实现包含 parent hash 的 `hash_block`。
5. 实现 `find_cached_prefix`，返回 hit block ids 和 hit tokens。
6. cache hit 后初始化 `request.block_table` 和 `request.num_computed_tokens`。
7. 改造 scheduler，让 prefill 从 `num_computed_tokens` 开始。
8. 改造 BatchBuilder，验证非零 start position 的 positions 不重置。
9. 实现 `register_cached_block`。
10. 实现 request 结束时只释放 active refs。
11. 实现 `evict_one_cached_block`，只淘汰没有 active request 的 cached block。
12. 增加 prefix cache hit tokens 指标。
13. 构造固定 system prompt workload，验证第二个请求 TTFT 下降。
14. 构造随机 prompt workload，验证 hit ratio 接近 0。
15. 写一段面试回答：为什么 prefix cache 需要 parent hash 和 extra hashes？

## 50.36 本章总结

prefix cache 是 paged KV cache 之后最自然的升级。

它的核心不是缓存字符串，也不是缓存最终答案，而是复用已经计算完成的 KV blocks。

第一版 hash-based prefix cache 可以很简单：只缓存 full blocks，用 parent hash 串起 prefix chain，用 extra hashes 保证模型、tokenizer、LoRA、多租户和多模态条件一致。

新请求命中 cache 后，本质上只是把 request 的 block table 前缀接到 cached blocks，并把 `num_computed_tokens` 设置为 hit tokens。后续 scheduler、BatchBuilder、OutputProcessor 仍然走原来的 suffix prefill 和 decode 流程。

prefix cache 的收益主要体现在 TTFT 和 prefill tokens/s，但代价是占用更多 KV blocks。因此必须有 ref count、eviction、水位线、hit token ratio、cached blocks 和 free blocks 等观测能力。

第一版不要急着实现 RadixAttention、partial block cache 或复杂 eviction。

先把 full-block hash cache、ref count、suffix prefill 和指标做正确，再继续升级。

下一章会进入第 51 章：实现 preemption、recompute 和 swap 的最小版本。
