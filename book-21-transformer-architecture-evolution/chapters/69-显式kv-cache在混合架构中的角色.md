# 第 69 章 显式 KV Cache 在混合架构中的角色：不是有了递归状态就能删掉 KV

## 69.1 混合模型为什么仍然需要显式 KV

递归或线性层可以用固定状态扫描历史，但显式 attention 仍然需要直接读取 K/V。一个 hybrid 模型可能同时拥有局部 KV、global KV、latent cache 和 recursive state。把它们统称为 cache，会隐藏它们在大小、读写、回滚和共享上的差异。

因此看到“模型使用递归 attention”时，不能直接推断“线上没有 KV cache”。只要仍有一部分层做显式 attention，长请求就会保留对应层的历史表示。

## 69.2 按层计算资源

对保存完整历史的层，KV 教学估算为：

```math
M_{\mathrm{KV},l}=2BLTH_{\mathrm{kv},l}d_{h,l}b_l
```

如果只有 global layer 集合 `L_g` 保存完整 KV：

```math
M_{\mathrm{KV,total}}=\sum_{l\in L_g}M_{\mathrm{KV},l}
```

局部窗口为 `w_l` 时，活跃长度可以替换为窗口或页数；递归 state 需要单独计算，不能把它直接当作 KV 的一个更小 dtype。

## 69.3 一个四层 global 的例子

假设模型有 32 层，只有 4 层保存全局 KV，其余层使用 local 或 state。理想情况下完整 KV payload 约是全层保存的八分之一。但如果这 4 层的 head 数更高、使用 FP16、还要保留 prefix metadata 和 workspace，实际比例可能更大。

更重要的是，这 4 层仍然决定长上下文的精确检索上限。容量规划若只看递归层固定状态，会高估并发；能力评估若只看递归层，又会漏掉 global path 的贡献。

## 69.4 KV 的生命周期

prefill 把输入写入 cache，decode 追加新 token，continuous batching 读取活跃页，完成/取消释放；prefix cache 可能共享相同前缀，preemption 可能 swap、重算或压缩。混合模型还要让每个层知道自己的 cache 类型。

每个对象至少需要 owner、request id、layer type、dtype、position range、page id、revision 和引用计数。不能只用一个 `past_key_values` 指针覆盖 local、global、latent 和 recursive state。

## 69.5 调度和 admission control

长请求进入时，scheduler 要估算最坏的 global KV、局部窗口、递归 state、临时 workspace 和输出长度。若只按输入 token 数 admission，可能低估某些层的峰值；若只看显存静态权重，又无法处理多个请求的增长。

一个粗略容量条件是：

```math
M_{\mathrm{weights}}+M_{\mathrm{KV}}+M_{\mathrm{state}}
+M_{\mathrm{workspace}}\le M_{\mathrm{usable\ GPU}}
```

真实服务还需给通信、碎片、CUDA graph 和其他 batch 留余量。

## 69.6 一个偶发 cache 错误的排查

先用单请求、短输入比较 full recompute 与 cache decode；再打开 batching、prefix sharing、preemption、量化和 speculative。每一步记录 layer id、position、page id、state step 和 checksum。

如果只有 batch 失败，查 page index、padding mask 和请求重排；如果只有恢复失败，查 snapshot 是否包含 global KV 和 recursive state；如果只有 speculative 失败，查 candidate commit/rollback 是否跨 cache 类型一致。

## 69.7 prefix cache 的边界

显式 KV prefix cache 可以共享固定 system prompt 或文档前缀，但 hybrid 模型还要决定 latent/state 是否也能共享。共享条件包括相同 model revision、tokenizer/template、position、dtype、layer schedule 和策略版本。某一层命中而另一层未命中时，scheduler 需要正确拼接，不应把部分命中误报成全命中。

prefix cache 还涉及租户权限。缓存中含有一个租户的文档，不能因为 token 前缀相同就跨租户复用。

## 69.8 评估维度

测 page allocation、prefix hit、last-block waste、state snapshot、preemption、FP8/低精度 KV、speculative rollback、长上下文 evidence recall 和 p99。报告每种 layer path 的显存，而不是只给总 KV GiB。

## 69.9 常见失败

以为递归 state 取代所有 KV；global layer 数未计入容量；层索引错位；local/global position 不一致；只释放 KV 未释放 state；prefix cache hash 不完整；跨请求共享；恢复后 cache step 错误；只测单请求 happy path。

## 69.10 面试回答与练习

回答“混合架构还要不要 KV cache”时，应说只要有显式 attention 层就需要对应 KV；递归/latent 路径改变的是部分历史表示和资源模型。容量要按层、cache 类型、窗口、dtype、state 和临时空间计算，生命周期要覆盖 batching、preemption、prefix sharing 和 rollback。

练习一：给定 32 层中 8 个 global layer，计算理想全层 KV 比例。

练习二：设计一个 batch 重排导致 KV 错位的回放。

练习三：列出 hybrid prefix cache 的共享条件。

### 69.10.1 按层计算显式 KV

设 hybrid 模型有 `L_e` 个显式 attention 层，每层 KV head 数为 `H_kv`，head dimension 为 `d_h`，序列长度为 `T`，元素字节数为 `b`，则显式 KV payload 可粗略写成：

```math
M_{\mathrm{KV,explicit}}
\approx2B L_e T H_{\mathrm{kv}}d_h b
```

递归或 latent 层还需要各自的 state，不能把 `L_e/L` 直接当作总显存比例。若 global 层的 head 或维度更大，真实比例可能高于层数比例。

### 69.10.2 cache 类型和调度器

allocator 要知道一个 request 同时拥有显式 KV、latent cache、recursive state 和临时 speculative state。preemption 时可选择保存显式 block、重算局部层或迁移 state；恢复时必须保证各类状态对应同一个 token prefix 和 position。

prefix sharing 只有在模型 revision、template、position/reset、dtype、tenant 和 cache layout 都一致时才安全。混合架构不能沿用只考虑 token 前缀的简单 cache key。

### 69.10.3 worked example：为什么少数 global 层仍昂贵

32 层模型中只有 8 层使用 global attention，看似只需四分之一显式 KV。但若这 8 层的 `H_kv d_h` 是其他层的两倍，并且每层需要额外通信 buffer，实际显式 payload 接近一半甚至更高。容量规划必须按层配置读取，而不是只看“8/32”。

### 69.10.4 质量验收条件

显式层往往承担精确检索或引用路径。量化、淘汰或减少这些层的 cache 时，应测数字复制、版本冲突、长文档引用和工具历史。性能优化如果只保持 perplexity，却损害关键证据召回，不能视为无损。

## 69.11 显式 KV 的价值不只是精度

显式 KV 还承担可回放、可检查和可迁移的作用。调试长上下文时，可以对某个层、某个 head 或某个位置做遮挡，观察 logits 如何改变；递归 state 或 latent cache 则更难直接定位原始证据。

因此混合架构即使把大部分历史交给 state，也可能保留少量显式 KV 作为精确证据通道、审计通道或 fallback。显式路径的比例不是越低越好，而要和任务风险、引用要求及可观测性一起决定。

## 69.12 KV budget 与 admission control

请求进入服务前，可以估算每个 global layer 的 KV 需求：

~~~math
M_i\approx 2B_iT_iH_{\mathrm{kv}}d_hb
~~~

再将请求需求和可用 block pool 比较。若没有足够预算，系统应拒绝、排队、缩短上下文、改走 retrieval 或选择更高压缩路径；不能在执行中随机丢掉某些关键页。

admission control 还要考虑未来 decode 长度。只按 prompt token 分配，可能在生成一半时耗尽 KV；只按 max_new_tokens 预留，又会降低并发。可采用可增长配额和可观测 preemption，但必须保证已提交状态和回滚状态一致。

## 69.13 显式 KV 与 prefix cache 的区别

请求内 KV 是某个请求当前前缀的执行状态；prefix cache 是可以被多个请求复用的共享结果。共享前必须验证 tokenizer、模板、模型 revision、position、权限和 dtype。即使 token 前缀相同，不同 system policy 或 adapter 也可能使隐藏状态不同。

混合架构中还要把 global KV、local window 和 recursive state 分开标注。只共享一类 cache 而遗漏另一类，会造成“命中率很高但输出错误”的隐蔽问题。

## 69.14 显式 KV 的压缩和回退边界

混合模型通常不会在所有层使用同一种历史表示。某些层保存完整 K/V，某些层保存滑窗，某些层保存递归 state。请求增长时，runtime 必须知道每层的逻辑位置和物理布局，不能用一个统一的 token offset 代替所有路径。

当显式 KV 预算不足时，回退策略要有顺序：先释放已完成请求的缓存，再尝试 prefix sharing 或重计算；如果任务允许，再减少输出预算或切换 retrieval；高风险任务则应拒绝而不是静默丢页。任何压缩或重算都要进入 trace。

## 69.15 KV 的精确回放实验

对同一 prefix 保存全量 KV，然后分别执行单请求 decode、continuous batching、preemption restore 和 speculative accept/reject。比较每层 K/V 的 shape、dtype、page id、logical position 和最终 logits。

再对一个 global layer 做单页遮挡。如果输出没有变化，可能该页没有被读取，也可能 query 路径失效；如果输出变化但引用没有变化，说明模型有冗余路径。遮挡实验能把 cache 存在和 cache 真正被使用区分开。

## 69.16 KV admission 是长上下文服务的门

KV cache 不是请求开始时一次性分配完就结束。scheduler 需要根据输入长度、预计输出、并发和层配置判断请求是否有资格进入 GPU。可增长的配额要防止生成中途耗尽，过度预留又会降低并发。

一个简化的 admission 条件是：

~~~math
M_{\mathrm{required}}
=M_{\mathrm{weights}}+M_{\mathrm{workspace}}
+M_{\mathrm{KV}}(T_{\mathrm{in}}+T_{\mathrm{planned\ out}})
\le M_{\mathrm{budget}}
~~~

估计错误时应有排队、缩短上下文、检索回读或拒绝路径，不能静默丢掉重要 cache page。

## 69.17 显式 KV 与压缩路径的职责

完整 KV 保留每个历史 token 的位置化 K/V，适合 query-time 精确访问；latent、state 和滑窗保存的是压缩或局部历史，适合降低内存。混合模型需要按层记录哪一种路径承担什么任务。

当 verifier 需要解释答案时，显式 KV 仍不能替代原文 provenance；当请求需要恢复时，latent/state 也不能只用 token offset 重建。cache、证据和状态是三个不同的对象。

## 69.18 KV 的淘汰、共享和安全

淘汰策略要考虑请求优先级、未完成输出、prefix sharing、租户权限和模型 revision。共享 prefix 前要验证模板、adapter、position、dtype 和 policy；相同 token 字符串不保证隐藏状态语义相同。

请求结束后要释放物理 page、索引和元数据。高风险系统还要确认被删除的 cache 不会从共享池、snapshot 或日志中重新读出。

## 69.19 从 cache 结构走向生产验证

显式 KV 在混合架构中仍是精确历史读取的基础。递归和 latent state 可以降低部分成本，却不能让 allocator、scheduler 和容量规划消失。生产验证应把“是否能缓存”拆成四个问题：分配是否正确、读取是否正确、释放是否安全、恢复是否可重复。

最小回归包含 full prefill 与 incremental decode 对照、不同 page/block 大小、长短请求混合、prefix sharing、抢占、取消、batch reorder 和模型灰度。对每个请求记录 cache owner、layer kind、position range、dtype、model revision 和 block table checksum；比较输出 token、KV bytes、命中率、碎片率、抢占恢复时间和跨租户读取。混合模型还要分别统计 global KV、local window、latent 和 recursive state，不能用一个平均 cache bytes 掩盖峰值。

容量规划应使用峰值而不是平均：

```math
M_{\mathrm{peak}}
=M_{\mathrm{weights}}+M_{\mathrm{KV,global}}
+M_{\mathrm{KV,local}}+M_{\mathrm{latent}}
+M_{\mathrm{state}}+M_{\mathrm{metadata}}+M_{\mathrm{workspace}}.
```

若 cache 校验失败，系统必须选择重算、显式路径或拒绝请求，并将原因写入 trace。模型层表和 cache 语义必须以具体实现为准，但 allocator、scheduler、租户隔离和故障回退是所有混合架构都绕不开的部署问题。

## 69.20 KV cache 保存的到底是什么

在 decoder-only Transformer 中，某一层的历史 token 会产生 key 和 value。增量 decode 时，过去 token 的 K/V 不再重复计算，当前 query 只需要与缓存中的 key 做匹配，再读取 value。对长度 L、batch B、KV head 数 H、head dimension d 和每元素字节数 s，粗略显存是：

~~~math
M_{\mathrm{KV}}
=2L B H d s.
~~~

这里的 2 是 K 和 V 两种张量，不是层数；完整模型还要乘以 layer 数，并加 page metadata、对齐、临时 workspace 和通信 buffer。把一个单层公式当成整机显存，是容量规划中最常见的低估。

## 69.21 混合架构为什么仍然需要显式 KV

递归 state 和 latent cache 能够压缩一部分历史，但显式 KV 仍有四个不可替代的作用：

1. 直接访问近期 token，保证局部语法和格式。
2. 为 global attention 提供可寻址的远程候选。
3. 在高风险任务中保留可回读的原子证据。
4. 作为 state/latent 路径失败后的 fallback。

因此，“混合架构减少 KV”更准确的说法是“减少某些层、某些窗口或某些路径的 KV”，而不是“KV 不再重要”。如果 global layer 的请求比例很低但每次都需要完整历史，少量显式路径仍可能主导峰值显存。

## 69.22 prefill 和 decode 的缓存差异

Prefill 一次处理长输入，可以并行计算 K/V；decode 每生成一个 token，就向缓存追加一列或一个 page。两阶段的瓶颈不同：

| 阶段 | 主要压力 | 常见优化 |
| --- | --- | --- |
| prefill | 矩阵计算、输入带宽、长 prompt | chunked prefill、FlashAttention、context parallel |
| decode | KV 读带宽、调度、page 访问 | paged cache、GQA/MLA、continuous batching |

混合架构还要记录不同层的 cache 类型。state 层可能只有固定状态，local 层保存窗口，global 层保存更多页面。调度器不能用一个统一的“每 token KV bytes”估计所有请求。

## 69.23 page allocator 的正确性

Paged KV cache 把逻辑序列切成固定大小的 page，再通过 block table 映射到物理显存。它解决的是碎片和动态增长问题，不改变模型需要的历史表示。一个逻辑 token 的地址至少依赖：

~~~text
request id
layer id
logical token index
page size
block table
model revision
~~~

混合架构中，local window 滑动会释放旧 page，global path 可能继续保留另一个 page 集合。若 allocator 只按请求释放、不按路径记录 owner，可能出现 use-after-free 或错误共享。

## 69.24 cache 命中不是字符串命中

Prefix cache 共享的前提不只是 token ids 相同，还包括：

1. tokenizer 和 special token 版本一致。
2. 模型权重、adapter、量化和 position 处理一致。
3. system prompt、权限域和 policy context 一致。
4. attention mask、segment/reset 和 layer schedule 一致。
5. cache page 没有被其他租户污染。

相同的可见文本在不同 adapter 或权限上下文下，hidden state 可能不同。高风险系统不能只用字符串 hash 作为共享 key。

## 69.25 显式 KV 的版本和事务

请求生成时，KV cache 不是普通只读文件。新 token 到来会追加写入，推测解码可能先写入候选，再因为 target 拒绝而回滚；beam 或多 Agent 分支可能共享前缀后各自追加。可以把状态分为：

~~~text
committed prefix
tentative suffix
checkpoint snapshot
released pages
~~~

只有被接受的 token 才能从 tentative 迁移到 committed。回滚时要恢复 page length、logical position、sampling state 和与混合路径相关的 state/latent。只删除输出文本而保留 KV，会造成下一次读取看见不存在的 token。

## 69.26 offload 与多级缓存

GPU KV、CPU KV、远端 KV 和重算之间存在明确 trade-off：

~~~math
T_{\mathrm{next}}
=T_{\mathrm{GPU\ read}}
+T_{\mathrm{PCIe/NVLink}}
+T_{\mathrm{remote}}
+T_{\mathrm{recompute}}.
~~~

短请求通常不值得把 KV 搬到远端；长请求可能宁可接受一次传输也不愿重新 prefill。混合架构可以把近期 local KV 留在 GPU，把低频 global page 放在 CPU 或外部存储，但访问模式必须可预测，否则 p99 会被偶发 page miss 拉高。

## 69.27 安全和租户隔离

KV 里面包含模型对输入的中间表示，仍可能携带敏感信息。需要：

1. 以租户和权限域隔离 page pool 或至少隔离分配和引用。
2. 取消请求后清理物理 page 和 block table。
3. 禁止跨 policy context 共享 prefix。
4. 监控 snapshot、日志和 crash dump 是否带出缓存。
5. 对缓存迁移和远端 offload 加密并记录审计。

“模型没有直接输出 secret”不等于 cache 没有泄露风险。安全边界应覆盖 allocator、snapshot 和运维工具。

## 69.28 一个容量估算例子

假设模型有 40 层，B=4，历史长度 L=32768，GQA 使用 8 个 KV head，每个 head dimension 为 128，BF16 每元素 2 字节，则：

~~~math
M_{\mathrm{KV}}
=40\times2\times32768\times4\times8\times128\times2
~~~

约为 5 GiB，尚未计入 page metadata 和其他激活。若把 KV head 从 8 降到 2，理想化缓存降到四分之一，但 query head、attention kernel 和质量不会自动保持不变。若混合架构只有一半层保存完整 KV，还要明确另一半层是否使用 state、local window 或 latent，而不能直接再乘一个 0.5。

## 69.29 评测显式 KV 的四条曲线

KV 优化应分别测：

1. cache bytes：每个请求、每层和每 token 的实际占用。
2. access latency：decode 读缓存的带宽和 page miss。
3. quality：精确检索、格式遵循、冲突版本和引用。
4. lifecycle：共享、抢占、迁移、恢复、回滚和释放。

压缩后质量下降却不影响短问答时，说明回归集不够；显存下降但 p99 上升时，说明 offload 或 page 访问成为瓶颈；单请求正常而多租户失败时，优先查共享 key 和 allocator，而不是模型参数。

## 69.30 显式 KV 仍然承担精确回读

在混合架构中，递归或 latent state 负责压缩历史，显式 KV 仍可能承担局部精确注意力、最近上下文、工具 schema 和格式约束。它的价值不只是“旧架构遗留”，而是为需要逐 token 对齐的部分提供可寻址证据。

因此 cache 设计要按层和表示类型分账：局部 KV 的长度、全局 KV 的共享、latent/state 的固定或压缩大小，以及 speculative 临时页。把所有表示加总成一个 `kv_cache_bytes` 会丢掉调度和回滚语义。

## 69.31 显式 KV 的淘汰与回退

当显存压力增加，可以优先淘汰可重算的局部 KV、降低 speculative 窗口或转入异步处理；但不能静默删除唯一的递归 state 或外部工具状态。每种 cache 都要标明可重算、可迁移、可共享和可丢弃属性。

一个请求因 KV 不足而回退到短上下文时，用户应知道遗漏范围；如果回退仍声称完整分析，就把资源失败变成事实错误。

## 69.32 端到端验证

比较显式 KV 开启、关闭、部分淘汰和跨 worker 迁移四种路径，验证 logits、输出、引用、工具事件、显存和 p99。对相同 prefix，cache hit 与 full recompute 的差异应落在明确数值容差内；否则先修 cache contract，再谈压缩收益。

## 69.33 小结

显式 KV 是混合架构中的精确历史通道、回退通道和 serving 基础设施。它的成本可以通过 GQA、MLA、窗口、分页、offload 和重算降低，但每种优化都改变了地址、带宽、事务或质量边界。完整的 KV 设计必须把数学缓存公式、页面分配、版本契约、租户隔离和端到端 p99 放在同一套测试中。
