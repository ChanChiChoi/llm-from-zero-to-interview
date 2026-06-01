# 第 28 章 RadixAttention 与 Prefix Sharing

上一章从整体上看了 SGLang Runtime：请求进入后会经过 API server、tokenization、scheduler、memory pool、RadixAttention、model runner、sampler、grammar backend 和 streaming output。

本章专门深入 SGLang 最有代表性的后端机制：RadixAttention。

先澄清一个容易误解的点：RadixAttention 不是一种新的 attention 数学公式，也不是替代 FlashAttention 的 GPU kernel。它是一套 runtime 级别的 KV cache 复用机制，用 radix tree 管理 token prefix 到 KV cache 的映射，让复杂 LLM 程序中的共享前缀可以自动复用。

一句话概括：

> RadixAttention 用压缩前缀树维护 token 序列和 KV cache 的关系，在新请求到来时做最长前缀匹配，只计算未命中的 suffix；请求结束或生成新 token 后，把有复用价值的 KV cache 保留在树上，并在显存紧张时按策略淘汰。

## 28.1 本章目标

读完本章，你应该能讲清：

1. Prefix sharing 为什么是复杂 LLM programs 的核心性能机会。
2. RadixAttention 和普通 prefix cache 的直觉差异。
3. Trie、radix tree、token prefix、KV cache value 分别是什么。
4. Longest prefix match、insert、split、evict 的基本流程。
5. RadixAttention 如何减少 prefill 计算。
6. 为什么 RadixAttention 必须和 memory pool、scheduler 协同。
7. 哪些 workload 最容易受益，哪些场景收益有限。
8. 面试中如何解释 RadixAttention。

## 28.2 Prefix sharing 是什么

Prefix sharing 指多个请求或多个 generation call 共享同一段 token 前缀。

例如两个用户问题共享同一份长文档：

```text
request A: [system prompt][long document][question A]
request B: [system prompt][long document][question B]
```

共享部分是：

```text
[system prompt][long document]
```

如果 request A 已经对这段前缀做过 prefill，生成了对应 KV cache，那么 request B 再从头 prefill 就浪费了。

更复杂的例子是 self-consistency：

```text
shared prompt: [problem statement][few-shot examples][question]
  -> sample answer 1
  -> sample answer 2
  -> sample answer 3
  -> sample answer 4
```

多个 sample 共享完全相同的 prompt，只是后续 decode 分支不同。

Tree-of-Thought 更明显：

```text
root prompt
  -> thought A
      -> continuation A1
      -> continuation A2
  -> thought B
      -> continuation B1
      -> continuation B2
```

这里不仅 root 共享，`thought A` 下的多个分支也共享更长前缀。

Prefix sharing 的本质是：

```text
多个 token 序列拥有共同开头。
```

RadixAttention 要做的事情是：

```text
让 runtime 自动发现这些共同开头，并复用对应 KV cache。
```

## 28.3 为什么 prefix sharing 能省 prefill

自回归 LLM 生成分两个阶段：prefill 和 decode。

Prefill 处理输入 prompt，生成每层 attention 的 K/V：

```text
prompt tokens -> transformer forward -> KV cache + first logits
```

如果两个请求前 8000 个 token 相同，那么这 8000 个 token 的 K/V 在相同模型、相同 tokenizer、相同位置和相同附加条件下也是相同的。

第二个请求可以直接复用这 8000 个 token 的 KV cache，只计算后面的不同部分。

例如：

```text
cached:      [A B C D E F]
new prompt:  [A B C D E F G H]

reuse:       [A B C D E F]
compute:                 [G H]
```

这会带来两个收益：

1. 减少 prefill 计算量。
2. 降低 TTFT。

如果系统同时有大量类似请求，还会提升整体 input tokens/s，因为 GPU 不再重复处理同一段前缀。

但要注意：prefix sharing 主要节省 prefill，不直接消除后续 decode 成本。每条分支生成自己的新 token 时，仍然要一轮一轮 decode。

## 28.4 为什么复杂 LLM 程序特别需要 prefix sharing

普通 chat completion 可能只有一次请求：

```text
prompt -> answer
```

复杂 LLM 程序经常是：

```text
prompt -> gen A -> tool -> gen B -> branch -> gen C1/C2/C3 -> merge -> gen D
```

这种程序有几个特点：

1. 多次调用模型。
2. 调用之间共享上下文。
3. 有分支和并行。
4. 有多轮历史。
5. 有固定工具说明、规则、few-shot examples。
6. 有搜索树或候选答案树。

如果 runtime 只看到一堆独立请求，它很难知道这些调用之间的共享关系。

SGLang 的思路是：frontend language 让程序结构更显式，backend runtime 用 RadixAttention 自动利用共享前缀。

这就是为什么 RadixAttention 不只是“缓存一个 prompt”，而是面向复杂程序的 cache reuse 机制。

## 28.5 普通 prefix cache 的局限

第 23 章讲过 vLLM 的 prefix caching：它通常基于 KV block hash 复用已经计算过的 full blocks。

这种方式对固定前缀非常有效，例如：

```text
[system prompt][document][question A]
[system prompt][document][question B]
```

但复杂 LLM programs 的共享模式经常不只是“很多请求共享一个长开头”，而是树状共享。

例如：

```text
S
  -> A
      -> A1
      -> A2
  -> B
      -> B1
      -> B2
```

如果只用完整 prompt 作为 key：

```text
cache[full_prompt] = KV
```

那么 `S`、`S+A`、`S+B` 这些中间共享段很难被自然表达。

如果用 block hash，可以复用线性前缀上的 full blocks，但仍然需要额外的数据结构和调度策略来更系统地表达多分支、多层级的共享结构。

RadixAttention 的核心差异是：它直接把已见过的 token 序列组织成一棵 prefix tree。

## 28.6 Trie 和 radix tree

要理解 RadixAttention，先理解 trie 和 radix tree。

Trie，也叫前缀树。它用树表示一组序列的共享前缀。

假设有三条 token 序列：

```text
A B C D
A B E F
A B E G
```

普通 trie 可以表示成：

```text
root
  -> A
    -> B
      -> C
        -> D
      -> E
        -> F
        -> G
```

每条边只放一个 token。

Radix tree 是压缩前缀树。它会把只有单一路径的节点压缩成一条边。

同样的序列可以表示成：

```text
root
  -> [A B]
    -> [C D]
    -> [E]
      -> [F]
      -> [G]
```

在 RadixAttention 中，边上的 label 不是字符，而是一段 token ids：

```text
edge label = [token_1, token_2, ..., token_k]
```

节点或边关联的是这段 token 对应的 KV cache 位置。

## 28.7 RadixAttention 的数据结构直觉

概念上，RadixAttention 维护一棵树：

```text
Radix tree:
  key: token prefix
  value: KV cache reference
```

更具体一点，每个节点可能包含：

1. 从父节点到当前节点的 token segment。
2. 这段 token 对应的 KV cache 引用。
3. 子节点映射。
4. 父节点指针。
5. 最近访问时间或 LRU 信息。
6. 当前是否被 active request 引用。
7. 这段 cache 占用的 token 数或 memory slot。

可以用伪结构表示：

```python
class RadixNode:
    token_segment: list[int]
    kv_refs: list[KVRef]
    children: dict[int, RadixNode]
    parent: RadixNode | None
    last_access_time: int
    ref_count: int
```

真实实现会更复杂，尤其要和 GPU memory pool、paged KV layout、并发调度结合。但学习时先抓住这个抽象就够了。

## 28.8 KV cache value 是什么

Radix tree 的 key 是 token prefix，value 不是最终文本答案，而是 KV cache。

KV cache 包含每层 transformer 对历史 token 计算出的 key/value。

概念形状类似：

```text
KV cache:
  layer 0: K/V for tokens
  layer 1: K/V for tokens
  ...
  layer L-1: K/V for tokens
```

在 runtime 中，value 通常不会是一整块 Python 对象里的大 tensor，而是对 memory pool 中 cache slots 或 pages 的引用。

可以理解成：

```text
token prefix [A B C]
  -> KV positions [slot_0, slot_1, slot_2]
```

这样 model runner 在执行后续 suffix 或 decode 时，就能根据这些位置读取历史 KV。

## 28.9 Longest prefix match

新请求到来时，RadixAttention 最重要的操作是最长前缀匹配。

输入：

```text
new prompt tokens = [A B C D E F]
```

已有树：

```text
root
  -> [A B C]
      -> [X Y]
      -> [D]
          -> [Z]
```

匹配过程：

```text
[A B C D E F]
  matches [A B C]
  then matches [D]
  stops before [E F]
```

命中前缀是：

```text
[A B C D]
```

未命中 suffix 是：

```text
[E F]
```

runtime 接下来只需要对 `[E F]` 做 prefill，并把新路径插入树中。

伪代码可以写成：

```python
def match_prefix(root, tokens):
    node = root
    pos = 0
    matched_nodes = []

    while pos < len(tokens):
        child = node.find_child_starting_with(tokens[pos])
        if child is None:
            break

        n = common_prefix_len(tokens[pos:], child.token_segment)
        if n == 0:
            break

        matched_nodes.append((child, n))
        pos += n

        if n < len(child.token_segment):
            break

        node = child

    return matched_nodes, pos
```

`pos` 就是命中的 token 数。

## 28.10 为什么需要 split

Radix tree 的关键操作之一是 split。

假设树里已有一条边：

```text
[A B C D]
```

现在新请求是：

```text
[A B X Y]
```

二者共享 `[A B]`，但从第三个 token 开始分叉。

如果要让后续请求复用 `[A B]`，树必须把原边拆开：

拆分前：

```text
root
  -> [A B C D]
```

拆分后：

```text
root
  -> [A B]
      -> [C D]
      -> [X Y]
```

这就是 radix tree 能表达中间共享前缀的原因。

普通“完整 key -> value”的 cache 无法自然做这种拆分。

## 28.11 Insert 流程

当一个请求完成 prefill 或生成了新 token 后，runtime 可以把新 token 序列插入 radix tree。

简化流程：

```text
1. 从 root 开始匹配 token 序列
2. 如果完全命中已有路径，更新访问信息
3. 如果在某条边中间分叉，split 这条边
4. 为未出现过的 suffix 创建新节点
5. 把 suffix 对应 KV cache 引用挂到新节点
6. 更新 LRU 或访问时间
```

伪代码：

```python
def insert(root, tokens, kv_refs):
    matched_nodes, pos = match_prefix(root, tokens)

    if pos == len(tokens):
        touch_path(matched_nodes)
        return

    parent = find_insert_parent(matched_nodes)
    suffix_tokens = tokens[pos:]
    suffix_kv_refs = kv_refs[pos:]

    new_node = RadixNode(
        token_segment=suffix_tokens,
        kv_refs=suffix_kv_refs,
        parent=parent,
    )
    parent.children[suffix_tokens[0]] = new_node
```

真实实现要处理边中间 split、KV refs 切片、并发访问、memory pool 引用计数等问题。这里的伪代码只表达核心直觉。

## 28.12 一个完整例子：多轮 chat

假设第一轮对话：

```text
S = system prompt
U1 = user first message
A1 = assistant first answer
```

第一次请求结束后，树里可能有：

```text
root
  -> [S U1 A1]
```

用户继续第二轮：

```text
S U1 A1 U2
```

RadixAttention 可以命中：

```text
[S U1 A1]
```

只需要 prefill：

```text
[U2]
```

插入后：

```text
root
  -> [S U1 A1]
      -> [U2 A2]
```

如果另一个用户或另一个分支共享同一个 system prompt，但第一条消息不同：

```text
S U1'
```

树可能 split 成：

```text
root
  -> [S]
      -> [U1 A1]
          -> [U2 A2]
      -> [U1']
```

这样 `[S]` 可以被多个会话复用。

## 28.13 一个完整例子：few-shot batch

Few-shot 评测常见格式：

```text
instruction
example 1
example 2
example 3
question X
```

如果对很多问题做同一套 few-shot prompt：

```text
P = instruction + examples

request 1: P + question 1
request 2: P + question 2
request 3: P + question 3
```

树结构是：

```text
root
  -> [P]
      -> [question 1 answer 1]
      -> [question 2 answer 2]
      -> [question 3 answer 3]
```

`P` 越长，RadixAttention 的收益越明显。

这类 workload 的特点是：

1. 共享前缀长。
2. 分支多。
3. 每个分支 suffix 相对短。
4. 请求之间格式稳定。

它非常适合 prefix sharing。

## 28.14 一个完整例子：self-consistency

Self-consistency 会对同一个问题采样多个答案，然后投票或选择。

结构是：

```text
P = problem + reasoning instruction

P -> answer sample 1
P -> answer sample 2
P -> answer sample 3
P -> answer sample 4
```

Radix tree：

```text
root
  -> [P]
      -> [sample 1]
      -> [sample 2]
      -> [sample 3]
      -> [sample 4]
```

如果没有 cache reuse，`P` 会被 prefill 多次。

有 RadixAttention 后，`P` 只需要计算一次，后续分支从同一个 KV prefix 开始 decode。

这也是 SGLang 论文和博客中强调复杂 prompting workload 的原因：这些 workload 的共享前缀不是偶然现象，而是算法结构本身带来的。

## 28.15 一个完整例子：Tree-of-Thought

Tree-of-Thought 是 RadixAttention 最直观的例子之一。

假设搜索树：

```text
root prompt
  -> thought A
      -> eval A
      -> expand A1
      -> expand A2
  -> thought B
      -> eval B
      -> expand B1
      -> expand B2
```

对应 token prefix：

```text
R
  -> A
      -> A1
      -> A2
  -> B
      -> B1
      -> B2
```

Radix tree 自然表达这棵搜索树：

```text
root
  -> [R]
      -> [A]
          -> [A1]
          -> [A2]
      -> [B]
          -> [B1]
          -> [B2]
```

每次扩展一个节点时，都可以复用从 root 到该节点路径上的 KV cache。

如果搜索树很深，这种复用会非常重要。

## 28.16 RadixAttention 和 attention kernel 的关系

名字里有 Attention，容易让人误以为 RadixAttention 是一种 attention 计算公式。

更准确的分层是：

```text
RadixAttention:
  管理 prefix -> KV cache 的复用关系

Attention backend:
  真正执行 attention 计算
```

一次请求命中 RadixAttention 后：

```text
matched prefix KV 已经存在
  -> model runner 只对 suffix 做 prefill
  -> attention backend 在计算 suffix 时读取 matched prefix 的 KV
```

所以 RadixAttention 不替代 FlashAttention、paged attention 或其他 kernel。

它是 runtime 级别的 cache 管理和调度机制。

## 28.17 和 memory pool 的协同

Radix tree 只记录逻辑关系还不够，KV cache 必须真实占用显存或其他存储。

Memory pool 负责：

1. 分配 KV cache slots。
2. 回收不再需要的 slots。
3. 维护 slot 是否被 active request 使用。
4. 支持 cache 被保留和复用。
5. 在显存不足时提供可淘汰对象。

RadixAttention 负责：

1. 根据 token prefix 找 cache。
2. 把新 token path 插入树。
3. 判断哪些 cache 有复用价值。
4. 选择哪些叶子或路径可以淘汰。

两者必须协同。

如果 radix tree 里有节点，但 memory pool 中对应 KV 已经被覆盖，就会读错 cache。

如果 memory pool 保留了 KV，但 radix tree 找不到对应 prefix，就无法复用。

所以每次 eviction 都要同时更新：

```text
radix tree metadata
memory pool allocation state
KV cache references
```

## 28.18 Ref count：哪些节点不能淘汰

一个缓存节点可能正在被请求使用。

例如请求 R 正在基于 `[A B C]` decode，那么 `[A B C]` 对应 KV 不能被淘汰。

因此节点或 KV slot 需要类似 ref count 的机制：

```text
ref_count > 0: active request 正在使用，不能 evict
ref_count = 0: 没有 active request，可作为 cache 保留，也可在显存紧张时 evict
```

请求命中 prefix 时，要增加引用：

```text
match prefix -> pin cache -> ref_count += 1
```

请求结束或不再需要该 prefix 时，要减少引用：

```text
finish request -> unpin cache -> ref_count -= 1
```

Eviction 只能选择 ref count 为 0 的节点或叶子。

这和普通缓存不同：普通缓存淘汰最多导致下次 miss；KV cache 如果淘汰了正在运行请求需要读取的内容，会导致错误生成甚至崩溃。

## 28.19 Eviction：为什么通常淘汰叶子

Radix tree 中，内部节点可能被很多子路径共享。

例如：

```text
root
  -> [P]
      -> [A]
      -> [B]
      -> [C]
```

如果淘汰 `[P]`，那么 `[A]`、`[B]`、`[C]` 的前缀也失效。

因此 eviction 通常更倾向于从叶子节点开始。

叶子节点代表某条路径的末端，淘汰它对其他共享路径影响较小。

简化策略：

```text
1. 找到 ref_count = 0 的叶子节点
2. 按 LRU 选择最久未使用的叶子
3. 释放它关联的 KV cache
4. 从 radix tree 删除该叶子
5. 如果父节点因此变成无用叶子，可递归清理或压缩
```

这不是唯一策略，但符合直觉：优先释放复用价值较低、影响范围较小的 cache。

## 28.20 LRU 为什么不总是最优

LRU 的直觉是：最近用过的 prefix 未来更可能再次被用到。

这在 chat、few-shot、agent loop 中常常有效。

但 LRU 不总是最优。

例如：

1. 一个很长的 prefix 虽然不常用，但每次命中都能省大量 prefill。
2. 一个很短的 prefix 经常命中，但节省 token 很少。
3. 某些业务有周期性流量，刚被淘汰后马上又来。
4. 多租户场景中，高频小租户可能挤掉低频大价值 cache。

更理想的策略可能考虑：

1. 最近访问时间。
2. prefix 长度。
3. 节省的 prefill tokens。
4. 节点子树共享程度。
5. 租户优先级。
6. 请求类型。

但策略越复杂，维护成本和调度开销越高。

所以工程上常见做法是先用简单可靠策略，再根据指标调优。

## 28.21 Scheduler 为什么要知道 cache hit length

RadixAttention 的命中结果会改变 scheduler 对请求成本的估计。

假设请求 prompt 长度是 12000 tokens。

如果没有命中：

```text
prefill cost = 12000 tokens
```

如果命中 10000 tokens：

```text
prefill cost = 2000 tokens
```

这两个请求对 GPU 的压力完全不同。

Scheduler 如果不知道 cache hit length，可能做出错误决定：

1. 把一个实际很便宜的请求挡在队列外。
2. 把一个实际很贵的请求放进 batch，导致 decode 抖动。
3. 错估 TTFT。
4. 错估 KV allocation。
5. 错过 cache locality 更好的调度机会。

因此 RadixAttention 不是一个独立字典，而是 scheduler 决策的一部分。

## 28.22 Cache-aware scheduling

Cache-aware scheduling 指 scheduler 在选择请求时考虑 cache 命中情况。

例如 waiting queue 中有两个请求：

```text
R1: prompt 12000 tokens, cache hit 10000 tokens, need prefill 2000
R2: prompt 4000 tokens, cache hit 0 tokens, need prefill 4000
```

如果只看 prompt length，R1 看起来更贵。

如果看 uncached suffix，R1 反而更便宜。

再比如：

```text
R3 和当前 hot prefix 共享很长前缀
R4 不共享任何 prefix
```

优先调度 R3 可能提升 cache hit 和整体吞吐。

但 cache-aware scheduling 也不能过度偏向 cache hit 请求，否则 cache miss 请求可能饥饿。

所以它本质上是折中：

```text
cache reuse
fairness
TTFT
TPOT
KV capacity
batch utilization
```

## 28.23 Token prefix 完全匹配，不是语义相似

RadixAttention 匹配的是 token prefix 完全一致。

不是字符串相似，更不是语义相似。

例如下面两段文本语义相同，但 token 序列可能不同：

```text
"You are a helpful assistant."
"You are a helpful assistant ."
```

空格、标点、chat template、special token 都会影响 token ids。

因此要提高 RadixAttention 命中率，需要保证：

1. prompt 模板稳定。
2. system prompt 稳定。
3. few-shot examples 顺序稳定。
4. 工具说明稳定。
5. RAG 文档拼接格式稳定。
6. tokenizer 和 chat template 一致。
7. 不要在共享前缀中插入时间戳、随机 id 等变化字段。

如果共享前缀每次都带一个不同 request id，cache 基本无法命中。

## 28.24 正确性边界

KV cache 能不能复用，不只取决于 token 一样。

还必须保证影响 forward 的条件一致。

常见条件包括：

1. 模型权重一致。
2. tokenizer 一致。
3. chat template 一致。
4. RoPE scaling 或 position 规则一致。
5. LoRA adapter 一致。
6. 多模态输入内容一致。
7. prompt adapter 或特殊 embedding 一致。
8. tenant 隔离策略允许共享。
9. dtype、量化和 KV layout 兼容。

如果这些条件不一致，却复用同一份 KV cache，结果可能是错误的。

工程上通常需要把这些条件编码进 cache key、cache namespace 或额外 metadata。

Radix tree 管 token prefix，但 prefix 所属的上下文环境也必须正确隔离。

## 28.25 多模态场景

多模态模型会让 prefix sharing 更复杂。

例如图像输入可能在文本里表现为：

```text
<image>
```

但真正送入模型的是视觉 encoder 产生的 embedding 或一组 image tokens。

如果两个请求都有 `<image>` 占位符，但图片不同，不能复用同一段视觉相关 KV。

因此多模态 cache 需要考虑：

1. 图片内容 hash。
2. 图片预处理参数。
3. vision encoder 版本。
4. image token 数量。
5. image token 插入位置。
6. 文本 token 与视觉 token 的组合方式。

可以复用的情况是：

```text
同一张图片 + 同一段文本前缀 + 同一模型处理流程
```

不能只看文本 placeholder 是否一样。

## 28.26 与 vLLM prefix caching 的对比

可以用一张表总结直觉差异。

| 维度 | vLLM prefix caching | SGLang RadixAttention |
| --- | --- | --- |
| 核心结构 | block hash / block table | radix tree / prefix tree |
| 关注点 | 复用已计算 KV blocks | 自动管理复杂 prefix sharing |
| 典型场景 | 大量请求共享固定前缀 | 多轮、分支、self-consistency、ToT、agent |
| 匹配方式 | 按 block hash 链命中 | longest prefix match |
| 共享形态 | 更偏线性前缀复用 | 更自然表达树状共享 |
| 淘汰对象 | cached blocks | radix tree 节点或叶子关联 KV |
| 与程序结构关系 | 相对弱 | 更强调 frontend/runtime 协同 |

这张表不是说谁绝对更好。

vLLM 的设计非常适合高吞吐通用 serving。

SGLang 的 RadixAttention 更强调复杂 LLM program 中的共享前缀自动复用。

真实系统会不断演进，边界也会变化。面试中要讲设计重心，不要把二者说成简单替代关系。

## 28.27 RadixAttention 不适合解决什么

RadixAttention 不是万能优化。

它不适合解决：

1. 完全不同 prompt 的重复计算。
2. decode 阶段每个 token 的所有成本。
3. 模型权重太大导致的显存不足。
4. attention kernel 本身慢的问题。
5. 网络延迟、tokenizer 慢、RAG 检索慢。
6. 语义相似但 token 不同的请求复用。
7. 需要返回完全相同答案的 result cache 场景。

如果请求之间几乎没有共享前缀，RadixAttention 命中率低，收益自然有限。

如果瓶颈在 decode TPOT，而不是 prefill TTFT，RadixAttention 的收益也可能不明显。

## 28.28 典型收益场景

RadixAttention 更适合这些 workload：

1. Few-shot evaluation。
2. Self-consistency。
3. Tree-of-Thought。
4. Multi-turn chat。
5. Agent trajectory。
6. Tool use with long tool descriptions。
7. RAG with shared documents。
8. Long-context question answering。
9. 多候选答案生成。
10. 结构化抽取中的固定 schema prompt。

共同特征是：

```text
共享前缀长 + 分支多 + 模板稳定 + 重复调用多
```

如果满足这些特征，RadixAttention 很可能明显降低 TTFT 和 prefill 成本。

## 28.29 常见性能现象

现象一：prefix cache hit rate 高，但吞吐提升不明显。

可能原因：

1. 命中的 prefix 很短。
2. 瓶颈在 decode，而不是 prefill。
3. grammar mask 或采样开销较大。
4. GPU 已经被 decode batch 占满。
5. cache 管理开销抵消了部分收益。

现象二：TTFT 下降明显，但 TPOT 没变化。

这是正常现象。RadixAttention 主要减少 prefill，TPOT 更多取决于 decode step、KV 读取、batch size 和 attention backend。

现象三：显存占用变高。

可能是 cache 保留了很多 prefix。需要看命中率、eviction、保留 token 数和实际节省 tokens。

现象四：命中率突然下降。

可能是 prompt template 改了、system prompt 注入了动态字段、tokenizer 版本变化、路由打散了 cache locality，或 cache 被频繁 eviction。

## 28.30 调试 RadixAttention 的思路

排查 RadixAttention 是否有效，可以看这些指标：

1. prefix cache hit rate。
2. matched prefix length。
3. saved prefill tokens。
4. radix tree node count。
5. cached token count。
6. eviction count。
7. active cached token count。
8. free KV slots。
9. TTFT p50/p95/p99。
10. prefill tokens/s。
11. decode tokens/s。

排查顺序可以是：

```text
确认 workload 是否有共享前缀
  -> 检查 tokenized prefix 是否完全一致
  -> 检查 cache namespace / tenant / LoRA 是否一致
  -> 检查是否被频繁 eviction
  -> 检查 scheduler 是否利用 hit length
  -> 对比 TTFT 和 prefill tokens/s
```

不要只看“开启了 RadixAttention”。开启不代表一定命中，命中也不代表一定提升端到端指标。

## 28.31 面试官会怎么问

问题一：RadixAttention 是什么？

回答要点：它是 SGLang runtime 的自动 KV cache 复用机制，用 radix tree 管理 token prefix 到 KV cache 的映射，新请求通过最长前缀匹配复用已有 KV，只对未命中 suffix 做 prefill。

问题二：RadixAttention 是 attention kernel 吗？

回答要点：不是。它不是新的 attention 数学公式，也不是 FlashAttention 替代品，而是 runtime 层的 prefix cache 管理机制。真正的 attention 计算仍由 attention backend/kernel 执行。

问题三：为什么用 radix tree？

回答要点：复杂 LLM programs 的共享前缀经常呈树状结构，radix tree 可以紧凑表示大量 token 序列的共享前缀，支持 longest prefix match、split、insert 和 eviction。

问题四：RadixAttention 主要优化什么指标？

回答要点：主要减少重复 prefill，降低 TTFT，提高 input token 处理效率和整体吞吐；对 TPOT 的影响通常是间接的。

问题五：什么情况下 RadixAttention 收益不大？

回答要点：请求之间没有共享前缀、共享前缀很短、prompt 模板不稳定、cache 被频繁淘汰、瓶颈主要在 decode 或网络/检索/tokenizer 时，收益会有限。

## 28.32 标准回答模板

如果面试官问“讲讲 SGLang 的 RadixAttention”，可以这样回答：

```text
RadixAttention 是 SGLang Runtime 中用于自动 KV cache 复用的机制。它不是一种新的 attention kernel，而是用 radix tree 管理 token prefix 到 KV cache 的映射。

复杂 LLM 程序里经常有多次 generation、分支、self-consistency、Tree-of-Thought、多轮 chat 和 agent trajectory，这些调用之间会共享大量 prompt 前缀。如果每次都从头 prefill，会重复计算同样的 KV cache。

RadixAttention 的做法是：请求 tokenize 后，runtime 在 radix tree 中做最长前缀匹配，找到已经缓存的 prefix KV，只对未命中的 suffix 做 prefill。请求生成的新 token 和对应 KV cache 会插入树中；如果新路径和已有路径只共享一部分，就 split 节点，让共享前缀成为单独节点。显存紧张时，系统会根据 ref count 和 LRU 等策略淘汰没有 active request 引用的叶子节点。

它的主要收益是减少 prefill 计算、降低 TTFT，并提升复杂 LLM program 的吞吐。它特别适合 few-shot、self-consistency、Tree-of-Thought、多轮对话、agent 和共享长文档 RAG 等场景。但它匹配的是 token prefix 完全一致，不是语义相似；如果 prompt 模板不稳定、共享前缀短或瓶颈在 decode，收益就会有限。
```

## 28.33 小练习

1. 给定三条 token 序列 `[A B C D]`、`[A B E F]`、`[A B E G]`，画出 trie 和 radix tree。
2. 解释 RadixAttention 为什么需要 split 节点。
3. 用 self-consistency 画一个 prefix sharing 树。
4. 用 Tree-of-Thought 画一个两层搜索树，并标出哪些 KV cache 可以复用。
5. 解释为什么 RadixAttention 主要优化 TTFT，而不是直接优化 TPOT。
6. 列出会导致 token prefix 无法命中的 5 个工程原因。
7. 设计一个 RadixAttention dashboard，至少包含 hit length、saved tokens、eviction 和 TTFT。

## 28.34 本章总结

RadixAttention 的核心不是改变 Transformer attention 的计算公式，而是在 serving runtime 中自动管理 prefix sharing。

它用 radix tree 表达 token 序列之间的共享前缀，用 longest prefix match 找到可复用 KV cache，只对未命中 suffix 做 prefill；当新路径出现时，通过 split 和 insert 维护树结构；当显存紧张时，通过 ref count、LRU 和叶子淘汰等策略释放 cache。

理解 RadixAttention 后，就能解释为什么 SGLang 特别适合复杂 LLM programs：这些程序天然包含多次 generation、分支、并行、多轮历史和共享上下文。下一章会继续深入 SGLang scheduler，看看 runtime 如何在 cache hit、prefill、decode、KV capacity 和公平性之间做调度取舍。
