# 第 23 章 Prefix Caching 与 Prompt Cache

上一章讲了 vLLM worker、executor 和 engine 架构：engine core 负责调度和状态，executor 负责执行派发，worker 和 model runner 负责把调度结果落到 GPU forward。

本章继续讲一个对 TTFT、显存和成本都非常重要的能力：Prefix Caching 与 Prompt Cache。

一句话概括：

> Prompt cache 是更宽泛的平台层缓存概念，关注复用 prompt 前缀处理结果；vLLM 的 prefix caching 是 runtime 内部基于 KV cache blocks 的前缀复用机制，核心是用 hash 找到已经计算过的 full blocks，并在保证隔离和正确性的前提下复用它们。

## 23.1 本章目标

读完本章，你应该能讲清：

1. Prompt cache、prefix cache、KV cache、result cache、semantic cache 的区别。
2. Prefix caching 为什么主要优化 TTFT，而不是 TPOT。
3. vLLM 为什么只缓存 full blocks。
4. block hash 为什么要包含 parent hash、block tokens 和 extra hashes。
5. 新请求命中 prefix cache 后，scheduler 和 KV cache manager 如何处理。
6. cache salt、LoRA、多模态输入、模型版本、tokenizer 对隔离和正确性的影响。
7. prefix cache 命中率低、显存占用高、误复用风险分别怎么排查。

## 23.2 先区分几类缓存

大模型推理里经常同时出现这些词：

1. prompt cache。
2. prefix cache。
3. KV cache。
4. result cache。
5. semantic cache。

它们不是一回事。

| 缓存类型 | 缓存内容 | 主要位置 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| Prompt cache | prompt 前缀处理结果 | 平台层或 runtime 层 | 降低 prefill 成本 | 前缀匹配和权限隔离 |
| Prefix cache | 共享前缀对应的 KV blocks | runtime 内部 | 降低 TTFT 和 prefill 计算 | 显存占用、错误复用 |
| KV cache | attention 的历史 Key/Value | runtime 内部 | 加速 decode | 显存增长和生命周期管理 |
| Result cache | 完整请求到完整答案 | 平台层 | 最快返回 | 个性化、过期、权限问题 |
| Semantic cache | 相似问题到答案或中间结果 | 平台层 | 减少近似重复计算 | 误命中导致答非所问 |

本章重点是 prompt cache 和 prefix cache。

可以先记住：

```text
Prompt cache 是目标和概念层。
Prefix caching 是 runtime 内部的一种具体实现方式。
```

## 23.3 Prompt Cache 是什么

Prompt cache 缓存的是 prompt 中可复用的前缀处理结果。

典型场景：

1. 固定 system prompt。
2. 固定工具说明。
3. 固定 RAG 文档。
4. 固定代码上下文。
5. 多轮对话复用历史上下文。
6. 多 Agent 共用同一背景材料。

例如：

```text
prefix: 你是企业客服助手。以下是产品手册全文：...
query A: 如何退款？
query B: 是否支持海外配送？
```

如果 prefix 很长，两个请求都从头 prefill 会浪费大量计算。

Prompt cache 的目标是：让第二个请求不要重复计算相同 prefix。

它可以有不同实现：

1. 直接缓存 tokenized prompt。
2. 缓存 prompt 到某种中间表示。
3. 缓存 runtime 内部 KV blocks。
4. 在平台层识别共享前缀，再让 runtime 复用。

vLLM 的 prefix caching 属于第三类：缓存并复用已经计算好的 KV cache blocks。

## 23.4 Prefix Caching 优化什么指标

Prefix caching 主要优化 TTFT。

TTFT 可以粗略拆成：

```text
TTFT = 排队时间 + 输入处理时间 + prefill 时间 + first token 采样时间 + 输出传输时间
```

Prefix cache 主要减少的是 prefill 时间。

如果一个请求有 8000 tokens prompt，其中前 7000 tokens 命中 prefix cache，那么本次真正需要 prefill 的可能只剩后 1000 tokens。

这会显著降低 first token 前的 GPU 计算量。

但它不一定明显改善 TPOT。

TPOT 主要发生在 decode 阶段：每轮根据已存在 KV cache 生成下一个 token。Prefix cache 让请求更快进入 decode，但 decode 每 token 的成本仍然取决于当前上下文长度、batch size、attention backend、KV 读取和采样开销。

所以面试中不要说“prefix cache 加速所有生成阶段”。更准确的说法是：

```text
Prefix cache 主要降低重复前缀的 prefill 计算，因此改善 TTFT 和吞吐；对 TPOT 的影响是间接的。
```

## 23.5 为什么不能只按字符串缓存

最朴素的想法是：把 prompt 字符串作为 key。

```python
cache[prompt_text] = computed_prefix
```

这在平台层做粗粒度缓存时有用，但对 runtime 内部 KV block 复用还不够。

原因一：模型看的是 tokens，不是字符串。

同一个字符串在不同 tokenizer、不同 chat template、不同 special token 配置下，可能得到不同 token ids。

原因二：KV 结果依赖模型版本和权重。

同样 token ids，在不同模型、不同 checkpoint、不同 LoRA adapter 下，KV cache 不能共享。

原因三：KV 结果依赖上下文位置。

即使某个 block 内 tokens 一样，它前面的 prefix 不同，attention 看到的历史不同，KV 对后续计算的含义也不同。

原因四：多模态输入不能只看占位 token。

图像、音频、视频可能在 token 序列里表现为 placeholder，但真正输入 embedding 取决于具体媒体内容。

原因五：多租户要隔离。

不同租户即使 prompt 一样，也不一定允许通过 cache 共享产生时延侧信道。

因此 vLLM 的 prefix caching 采用 hash-based KV block 复用，而不是简单字符串缓存。

## 23.6 vLLM 的核心做法：缓存 full KV blocks

vLLM prefix caching 的核心思想是：

```text
把已经计算过的 KV cache full blocks 缓存起来；新请求到来时，如果前缀 tokens 对应的 block hash 命中，就直接复用这些 physical blocks。
```

为什么只缓存 full blocks？

因为 full block 边界稳定，hash 和复用更简单。

假设 block size 为 4：

```text
tokens: A B C D E F G
blocks: [A B C D] [E F G _]
```

第一个 block 是 full block，可以缓存。

第二个 block 还没满，通常不作为 prefix cache entry。

这样做的好处：

1. hash key 和 block 内容一一对应。
2. block table 更容易复用。
3. cache hit 只发生在稳定边界。
4. 避免 partial block 频繁变化带来的复杂性。

代价是：如果共享前缀长度不是 block size 的整数倍，最后不足一个 block 的部分不能直接命中，需要重新计算。

例如 block size 为 16，两个请求共享前 1000 tokens，那么最多只能命中前 992 tokens，剩下 8 tokens 要重新 prefill。

## 23.7 block hash 为什么需要 parent hash

vLLM 文档中，block hash 的核心组成包括：

1. parent hash。
2. block tokens。
3. extra hashes。

简化公式：

```text
block_hash_i = hash(parent_hash_i, block_tokens_i, extra_hashes_i)
```

parent hash 是前一个 block 的 hash。

这样 block hash 就包含了从开头到当前 block 的整条 prefix 信息。

为什么不能只 hash 当前 block tokens？

看这个例子：

```text
request A: X Y Z + [hello world]
request B: P Q R + [hello world]
```

当前 block `[hello world]` 一样，但前文不同。

在 causal attention 中，后面 token 可以 attend 到前面 token。即使当前 block tokens 一样，它们处在不同上下文之后，后续计算不能把两者简单当成同一个前缀结果。

parent hash 的作用就是把“当前 block 属于哪条 prefix 链”编码进去。

这很像链式哈希：

```text
block 0 hash = hash(None, block 0 tokens, extra)
block 1 hash = hash(block 0 hash, block 1 tokens, extra)
block 2 hash = hash(block 1 hash, block 2 tokens, extra)
```

所以新请求要命中 block 2，必须 block 0、block 1、block 2 组成的整条前缀链都一致。

## 23.8 extra hashes 包含什么

`extra_hashes` 用来编码 tokens 之外、但会影响 KV 正确性的因素。

常见包括：

1. 模型版本或模型标识。
2. tokenizer 或 chat template 版本。
3. LoRA adapter id。
4. 多模态输入 hash。
5. cache salt。
6. 特殊 prompt adapter 或任务配置。
7. 其他会改变 forward 结果的条件。

LoRA 为什么要进入 hash？

因为 LoRA 会改变模型权重。同样 token 在不同 LoRA adapter 下计算出的 K/V 不同。

多模态输入为什么要进入 hash？

因为 placeholder tokens 可能一样，但图片内容不同。只有把 image hash 加进去，才能避免不同图片复用同一段视觉相关 KV。

cache salt 为什么要进入 hash？

因为多租户场景下，即使内容相同，也可能不希望跨租户共享 cache。salt 能把 cache reuse 限制在同一个信任域里。

## 23.9 新请求命中 prefix cache 的流程

一个新请求进入 scheduler 后，KV cache manager 可以先尝试找已经计算过的 blocks。

简化流程：

```python
def find_cached_prefix(input_ids, extra_hashes, block_size, cached_blocks):
    parent_hash = None
    hit_blocks = []

    for block_tokens in split_full_blocks(input_ids, block_size):
        block_hash = hash_block(parent_hash, block_tokens, extra_hashes)
        block = cached_blocks.get(block_hash)
        if block is None:
            break
        hit_blocks.append(block)
        parent_hash = block_hash

    return hit_blocks
```

命中后，系统需要做几件事：

1. 把命中的 physical blocks 加入请求的 block table。
2. 增加这些 blocks 的 ref count。
3. 如果命中 block 当前在 free queue 中，需要 touch 它，避免被淘汰。
4. 对未命中的剩余 tokens 分配新 blocks。
5. 只对未命中部分执行 prefill。

简化状态变化：

```text
input tokens: [cached prefix][uncached suffix]
KV blocks:    [reuse blocks ][new blocks     ]
compute:       skip prefill   run prefill
```

这就是为什么 prefix caching 能降低 TTFT。

## 23.10 touch：命中后不能马上被淘汰

vLLM prefix caching 文档里有一个重要操作：touch。

当一个请求命中了 cached block，而这个 block 当前没有 active request 使用时，它可能在 free queue 中。

如果不 touch，后续分配新 block 时可能把它从 free queue head 弹出并 evict，导致当前请求刚命中就失效。

touch 要做的事情是：

1. 增加 ref count。
2. 如果 block 在 free queue 中，把它移出 free queue。
3. 标记它正在被 active request 使用。

伪代码：

```python
def touch_cached_block(block):
    if block.ref_cnt == 0:
        free_queue.remove(block)
    block.ref_cnt += 1
```

这说明 prefix cache 不只是 hash map 查找，还必须和 block 生命周期、free queue 和 ref count 一起工作。

## 23.11 请求结束后为什么 cache 还在

普通 KV cache 生命周期是：

```text
请求开始 -> 分配 KV -> 请求结束 -> 释放 KV
```

启用 prefix cache 后，生命周期变成：

```text
请求开始 -> 分配 KV -> full blocks 加入 cache -> 请求结束 -> ref count 归零 -> 进入 free queue 但保留 hash 和内容 -> 后续可能被复用或淘汰
```

所以请求结束后，某些 blocks 可能处于：

```text
cached but free
```

它们没有 active request 引用，可以被分配器拿去复用显存；但在被真正 evict 之前，hash map 仍能找到它们，后续请求可以复用。

这就是 prefix cache 的显存 trade-off：

1. 保留更多 cached blocks，命中率可能更高。
2. 但可用 free blocks 看起来更少，或者更容易触发 eviction/preemption。
3. 命中率低时，prefix cache 可能变成显存负担。

## 23.12 Eviction：缓存不是永久的

当 free blocks 不够时，系统需要淘汰 cached blocks。

常见策略是 LRU。

LRU 的直觉：越久没用的 cache，越可能未来也用不到。

简化流程：

```python
def evict_one_block():
    block = free_queue.pop_head()
    if block.block_hash is not None:
        cached_blocks.pop(block.block_hash, None)
        block.block_hash = None
    return block
```

注意：eviction 不应该发生在 ref count 大于 0 的 block 上。

如果一个 block 仍被 active request 引用，evict 会导致正在运行的请求读错 KV。

所以 cached block 可以分成三类：

1. active cached block：正在被请求使用，不能淘汰。
2. free cached block：没有请求使用，可以在需要时淘汰。
3. evicted block：hash 被移除，内容可被覆盖，不能再命中。

## 23.13 Prefix cache 和 continuous batching 的关系

Prefix cache 不是单独存在的优化，它会影响 scheduler。

如果一个 waiting 请求命中了大段 prefix，它的实际 prefill token 数变小。

例如：

```text
prompt length = 8192
cached prefix = 7168
uncached suffix = 1024
```

对 scheduler 来说，这个请求本轮 prefill 成本更接近 1024 tokens，而不是 8192 tokens。

因此 prefix cache 会影响：

1. token budget 估算。
2. KV block allocation。
3. waiting queue admission。
4. TTFT 预测。
5. chunked prefill 是否需要切分。

如果 scheduler 不知道 cache hit length，就会错误估计本轮工作量。

这也是为什么 prefix cache 通常内嵌在 KV cache manager 和 scheduler 交互中，而不是一个外部字典就能解决。

## 23.14 Prefix cache 和 Prompt Cache 的区别

现在可以正式对比。

| 对比点 | Prompt Cache | Prefix Caching |
| --- | --- | --- |
| 层次 | 平台层概念，可由多种方式实现 | runtime 内部机制 |
| 缓存对象 | prompt 前缀处理结果 | KV cache full blocks |
| 命中依据 | 文本、token 或业务 key | block hash 链 |
| 主要收益 | 减少重复 prefill | 减少重复 prefill |
| 管理资源 | 可能是内存、磁盘、远端缓存 | GPU/CPU KV blocks |
| 正确性要求 | 模型、tokenizer、权限一致 | block tokens、parent hash、extra hashes 一致 |
| 风险 | 业务隔离、过期、误匹配 | 显存占用、hash 冲突、跨租户泄露 |

可以这样理解：

```text
Prompt cache 是“我要复用 prompt 前缀”的产品/平台抽象。
Prefix caching 是“我用 KV blocks 复用前缀”的 runtime 实现。
```

## 23.15 多模态 prefix cache 的特殊问题

多模态请求通常会把图片、音频或视频变成 placeholder tokens 或 embedding。

例如：

```text
tokenized prompt: [text tokens, <image_placeholder>, <image_placeholder>, ...]
```

如果只 hash token ids，不同图片可能得到一样的 placeholder token 序列。

这会导致严重错误：请求 A 的图片 KV 被请求 B 的图片复用。

因此多模态 prefix cache 必须把媒体内容或处理结果的 hash 加入 extra hashes。

例如：

```text
extra_hashes = [image_hash, lora_id, cache_salt]
```

同时还要保证：

1. 图像预处理版本一致。
2. vision encoder 版本一致。
3. placeholder 数量和位置一致。
4. 多模态 embedding 注入位置一致。

否则即使文本 tokens 一致，也不能安全复用。

## 23.16 cache salt 与多租户安全

Prefix cache 的一个隐蔽风险是时延侧信道。

假设跨租户共享 cache，攻击者可以构造某个 prompt，然后观察 TTFT 是否显著变低，从而推测其他租户是否请求过相同内容。

cache salt 的作用是把 cache reuse 限制在同一个 trust domain。

例如：

```text
tenant A salt = salt_A
tenant B salt = salt_B
```

即使两个租户的 prompt tokens 完全一样，因为 salt 不同，block hash 也不同，不能互相命中。

这会降低跨租户 cache 命中率，但提升隐私和隔离。

多租户场景下，正确策略通常是：

1. 同一租户内部可以共享。
2. 同一企业 workspace 内可配置共享。
3. 不同租户默认隔离。
4. 高敏业务禁用跨用户 prefix cache。
5. cache hit 相关指标不要暴露给不可信用户。

## 23.17 Hash 算法和碰撞风险

vLLM 文档提到，新版本默认使用更安全的 hash 算法来降低碰撞风险，并允许配置不同 prefix caching hash algorithm。

从工程角度，hash 选择有 trade-off：

1. 加密强度更高的 hash 更安全，但可能更慢。
2. 非加密 hash 更快，但理论碰撞风险更高。
3. 跨语言、跨版本可复现序列化有利于分布式 cache。
4. Python pickle 类序列化可能不保证跨版本完全稳定。

如果 hash 碰撞导致错误命中，后果不是普通缓存返回旧值，而是模型可能基于错误 KV 继续生成，问题很难定位。

所以在多租户或安全敏感场景中，不能只追求 hash 速度。

## 23.18 Hybrid attention 下的 prefix cache

普通 full attention 模型中，prefix cache 检查比较直观：从左到右扫描 full blocks，遇到第一个 miss 就停止。

但混合 attention 模型更复杂。

例如 sliding window attention 只关注最近窗口内 tokens，不一定需要保留所有历史 KV。

Hybrid KV Cache Manager 文档中提到，不同 attention 类型有不同 prefix cache 规则：

1. full attention：cache hit prefix 要求所有历史 tokens 的相关 blocks 都存在。
2. sliding window：可能只要求窗口内最近 tokens 的 blocks 存在。
3. full + sliding window：需要对不同 group 的 cache hit 做交集。

简化理解：

```text
Full attention group hit length = A
Sliding window group hit length = B
Final reusable prefix length = intersection(A, B)
```

这说明 prefix caching 不是只和 tokens 有关，还和模型 attention 模式有关。

未来模型结构越复杂，KV cache group 和 prefix cache rule 也会越复杂。

## 23.19 命中率为什么低

线上 prefix cache 命中率低，常见原因包括：

1. prompt 实际不共享。
2. chat template 每次插入不同时间戳、request id 或随机字段。
3. RAG 文档排序不稳定。
4. tokenizer 或 special token 配置不一致。
5. block size 导致共享前缀没有落在 full block 边界。
6. LoRA adapter 不同。
7. cache salt 过细，隔离粒度太小。
8. 多模态 hash 不一致。
9. cache 容量太小，刚缓存就被 evict。
10. 流量分散在多个 replica，单实例看不到足够重复。

排查时不要只看业务说“我们的 system prompt 一样”。要看 tokenized prompt 的前缀是否真的一致。

建议指标：

1. prefix cache hit rate。
2. hit prefix length 分布。
3. saved prefill tokens。
4. evicted cached blocks。
5. cached-but-free blocks。
6. full block cache entries。
7. cache hit by tenant/model/route。
8. TTFT by hit length bucket。

## 23.20 Prefix cache 的副作用

Prefix cache 不是免费午餐。

副作用一：占用 KV memory。

请求结束后保留 blocks 可以提高命中率，但也会占用 cache pool，导致新请求更容易触发 eviction 或 preemption。

副作用二：增加管理复杂度。

需要维护 hash、ref count、free queue、LRU、touch、evict 和 request block table。

副作用三：隔离风险。

如果没有 cache salt 或租户隔离，可能产生跨租户侧信道。

副作用四：调试困难。

错误复用 KV 后，模型输出可能只是“变奇怪”，不一定立刻报错。

副作用五：命中率依赖流量模式。

如果请求前缀高度随机，prefix cache 带来的收益很小，却仍有管理开销。

因此 prefix cache 最适合共享前缀明显、长 prompt 较多、TTFT 敏感的业务。

## 23.21 什么时候应该启用或重视 Prefix cache

适合场景：

1. 长 system prompt。
2. 长工具说明。
3. RAG 中固定文档或固定知识库版本。
4. 代码助手中的固定仓库上下文。
5. 多轮对话中历史上下文重复出现。
6. Agent 系统中多个子任务共享背景。
7. 多用户访问同一公开文档。

不太适合场景：

1. prompt 很短。
2. 每个请求前缀都高度个性化。
3. 租户隔离要求极强，不能共享。
4. 显存非常紧张，cached blocks 挤占 active requests。
5. TTFT 主要瓶颈在排队、tokenizer、网络或上游 RAG，而不是 prefill。

## 23.22 面试官会怎么问

问题一：Prompt cache 和 prefix cache 有什么区别？

回答要点：prompt cache 是更宽泛的平台层概念，缓存 prompt 前缀处理结果；prefix cache 是 runtime 内部常见实现，复用前缀对应的 KV cache blocks。

问题二：vLLM prefix caching 为什么只缓存 full blocks？

回答要点：full block 边界稳定，hash 和复用简单，block table 易管理；partial block 频繁变化，复用和正确性处理更复杂。

问题三：block hash 为什么要包含 parent hash？

回答要点：同一个 block tokens 出现在不同前文后，attention 上下文不同，不能共享。parent hash 把整条 prefix 链编码进当前 block hash。

问题四：extra hashes 要包含什么？

回答要点：包含 LoRA id、多模态输入 hash、cache salt、模型/tokenizer/特殊配置等会影响 forward 结果或隔离边界的因素。

问题五：prefix cache 如何影响 scheduler？

回答要点：命中后实际 prefill token 数减少，scheduler 的 token budget、KV block allocation、admission control 和 TTFT 估算都要基于 uncached suffix，而不是完整 prompt。

问题六：多租户 prefix cache 有什么风险？

回答要点：可能通过 TTFT 差异形成侧信道，推测其他租户是否请求过相同内容；需要 cache salt、租户隔离和指标访问控制。

## 23.23 标准回答模板

如果面试官问“vLLM 的 prefix caching 是什么”，可以这样回答：

```text
Prefix caching 是 vLLM 用来复用相同 prompt 前缀 KV cache 的机制，主要目标是减少重复 prefill 计算，从而降低 TTFT 和提高吞吐。

它和更宽泛的 prompt cache 相关，但更偏 runtime 内部实现。vLLM 会把 KV cache 切成固定大小 blocks，只缓存已经填满的 full blocks。每个 cached block 会有一个 hash，这个 hash 通常由 parent hash、当前 block tokens 和 extra hashes 构成。parent hash 保证当前 block 属于同一条前缀链，extra hashes 用来编码 LoRA、多模态输入、cache salt 等会影响正确性和隔离的因素。

新请求进入时，KV cache manager 会从左到右计算 prompt blocks 的 hash，查找已经计算过的 blocks。命中的 blocks 会被加入请求 block table，ref count 增加，并从 free queue 中 touch 出来，避免被淘汰。未命中的后缀才需要执行 prefill。

它的收益是减少重复长前缀的 prefill，改善 TTFT；代价是占用 KV memory，并引入 hash、ref count、LRU eviction、多租户隔离和错误复用风险。
```

## 23.24 小练习

1. 假设 block size 为 16，两个请求共享前 1000 tokens，最多能直接命中多少 tokens？为什么？
2. 设计一个 block hash，要求支持 LoRA、多模态图片和多租户隔离。
3. 画出一个 cached block 从 active 到 free queue，再到被新请求 touch 的状态变化。
4. 解释为什么只按 prompt 文本字符串做 cache key 不足以保证 KV cache 复用正确。
5. 列出 prefix cache 命中率低的 8 个可能原因。
6. 设计一个 dashboard，用来观察 prefix cache 对 TTFT 的影响。

## 23.25 本章总结

Prompt cache 是大模型推理平台中复用 prompt 前缀处理结果的宽泛概念；prefix caching 是 vLLM-like runtime 中基于 KV cache blocks 的具体实现。

vLLM prefix caching 的关键设计是：只缓存 full blocks，用 parent hash、block tokens 和 extra hashes 构造 block hash；新请求命中后复用 physical KV blocks，只对未命中 suffix 执行 prefill。为了保证生命周期正确，还必须维护 ref count、free queue、touch、eviction 和 cached-but-free 状态。

Prefix cache 能显著改善共享长前缀场景下的 TTFT 和 prefill 吞吐，但它也带来显存占用、隔离、安全、hash 碰撞、命中率波动和调试复杂度。真正线上调优时，要结合业务 prompt 结构、租户边界、block size、cache 容量和 TTFT 分桶指标一起看。

下一章会继续讲 Tensor Parallel、Pipeline Parallel 与 serving 并行，重点解释 TP/PP/DP/EP 如何影响 worker 数量、显存分布、通信开销、吞吐和延迟。
