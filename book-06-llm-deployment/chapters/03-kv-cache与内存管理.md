# 第三章 KV Cache 与内存管理：把每个 token 的状态放在哪里

上一章说明了 decode 为什么会随着上下文变长而变慢。本章继续追问：模型究竟保存了什么，保存了多少，为什么同一个模型在不同并发和长度分布下会有完全不同的显存表现，以及推理引擎如何在请求不断加入和结束的情况下管理这些状态。

初学者可以先记住一句话：自回归生成时，历史 token 的 key 和 value 已经算过了，后续步骤可以复用它们；这份被复用的历史状态就是 KV Cache。它用显存换掉重复计算，但不会让历史上下文从计算中消失，新 token 仍然要用 query 访问历史 K/V。

专家需要再拆开四个问题。第一，cache 的 shape 由层数、K/V head 数、head dimension、上下文长度和数据类型共同决定。第二，MHA、MQA、GQA，甚至 MLA 或混合状态架构，保存的东西并不相同。第三，连续显存预留会产生碎片和空闲槽，PagedAttention 一类方法解决的是地址映射和分配问题，不是改变 attention 数学。第四，量化、prefix 复用、淘汰和跨 GPU 拆分都必须同时考虑质量、隐私、通信和恢复语义。

本章建立 KV Cache 的内存与生命周期模型。第 2 章已经解释 prefill/decode 和指标口径，第 6 章会进一步讨论 batching 与调度，第 7 章讨论模型和 cache 的量化，第 8 章讨论并行推理；这里不把它们的实现细节提前写成另一份手册，而是把后续章节共同依赖的账算清楚。

## 3.1 KV Cache 为什么存在

### 3.1.1 Self-Attention 中的 K、V

对某一层的隐藏状态 X，self-attention 通常先计算：

~~~math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
~~~

当前 query 会和历史 key 做相似度计算，再用得到的权重聚合历史 value：

~~~math
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(
\frac{QK^\mathsf{T}}{\sqrt{D_h}}+M_{\mathrm{causal}}
\right)V.
~~~

在 decode 的第 t 步，新 token 的 query 需要看到从输入到当前为止的上下文。历史 token 的 key/value 在它们自己的位置上已经由固定模型参数计算完成；新 token 加入之后，历史 token 的 K/V 不会改变。因此可以把历史 K/V 放在 GPU 显存中，下一步直接读取。

如果没有 cache，长度为 T_ctx 的上下文在每一个新 token 到来时都要重新计算历史 token 的 K/V。若有 cache，每一步只为新 token 计算新的 K/V，再追加到状态中。节省的是历史 K/V 的重复投影和写入，并不是所有 attention 工作。

### 3.1.2 Cache 保存的不是 logits

KV Cache 保存的是各层 attention 的 K 和 V，通常不保存完整的 logits 序列。logits 是当前最后位置对整个词表的输出，生成策略选出一个 token 后，下一步的模型状态会继续变化；历史 token 的 K/V 才是可以稳定复用的中间结果。

这一区分很重要：

1. 保存 logits 不能替代下一步各层的 attention 和 MLP 计算。
2. 保存 K/V 需要为每一层、每个活跃请求保留状态。
3. 保存的 K/V 与位置编码、模型 revision、dtype 和 attention 实现有关。
4. 改变 chat template 或位置处理方式，可能使旧 cache 不能复用。

不同模型和 runtime 的 Python 对象名字可能是 past_key_values、cache 或内部 block pool，但需要核对的语义都是“每层历史 attention 状态的存储与读取”。

### 3.1.3 从空状态到释放

一个请求的 cache 生命周期可以画成：

~~~text
empty
  -> prefill allocation and write
  -> decode append and read
  -> completed / cancelled / failed
  -> release, prefix reuse, offload, or eviction
~~~

请求完成后，属于该请求的私有状态必须及时释放；如果它的前缀被允许进入可复用 cache，则只能保留经过权限和版本校验的公共部分。客户端断开、服务超时和工具调用失败都必须触发同样的清理语义。很多所谓“KV Cache 泄漏”并不是 CUDA 内核永远不释放，而是异常路径没有把请求从 active table、block table 或 prefix cache 引用中移除。

## 3.2 Cache 的形状和单位 token 账本

### 3.2.1 标准 MHA 的形状

设模型有 L 层，第 l 层有 H_kv 个 K/V head，每个 head 的维度为 D_h，当前有 B 个请求，每条序列长度为 T_ctx。一层的 K 和 V 可以表示为：

~~~math
K_l,V_l\in
\mathbb{R}^{B\times H_{\mathrm{kv}}\times T_{\mathrm{ctx}}\times D_h}.
~~~

两个张量各占：

~~~math
B\cdot H_{\mathrm{kv}}\cdot T_{\mathrm{ctx}}\cdot D_h\cdot b
\quad\text{bytes}.
~~~

把 K、V 和所有层相加，得到同长度、同 dtype 的简化估算：

~~~math
M_{\mathrm{KV}}
\approx
2LB T_{\mathrm{ctx}}H_{\mathrm{kv}}D_h b.
~~~

其中 `B\geq0`、`T_ctx\geq0`，`L,H_kv,D_h` 为正整数，`b>0` 是每个元素的字节数，例如 FP16 和 BF16 通常是 2，FP8 或 INT8 的数据部分通常是 1。这个式子没有计入 block table、scale、对齐、复制、通信 buffer 和 allocator 元数据，所以它是容量模型的基础，不是最终显存读数；空活跃 batch 的理论 cache 可以是零，但不能据此推出服务已经完成了一次有效 decode。

### 3.2.2 变长请求必须按总长度相加

线上 batch 很少是每条请求恰好同样长。第 i 条请求的上下文长度为 T_ctx,i 时，所有活跃请求的状态大小应写成：

~~~math
S_{\mathrm{ctx}}=\sum_{i=1}^{B_{\mathrm{active}}}T_{\mathrm{ctx},i},
~~~

~~~math
M_{\mathrm{KV}}
\approx
2L H_{\mathrm{kv}}D_h b\,S_{\mathrm{ctx}}.
~~~

不要把 B_active * T_max 当成真实使用量，除非 runtime 确实按最大长度为每条请求完整预留。连续 batching、paged allocation 和 ragged kernel 往往按真实 token 数管理，但仍可能有 block rounding 和 padding 浪费。

单个上下文 token 在所有层的理论 cache 成本是：

~~~math
M_{\mathrm{token}}
\approx
2L H_{\mathrm{kv}}D_h b.
~~~

这个量要求 `L,H_kv,D_h` 为正且 `b>0`。它特别适合做容量规划。若可供 KV Cache 使用的显存预算为 `M_budget>0`，并且假定所有请求上下文长度都是 `T_ctx>0`，粗略并发上限为：

~~~math
B_{\mathrm{rough}}
\approx
\left\lfloor
\frac{M_{\mathrm{budget}}}
{2L T_{\mathrm{ctx}}H_{\mathrm{kv}}D_h b}
\right\rfloor.
~~~

`M_budget` 必须是扣除权重、workspace、激活、通信和安全余量后的正预算。这个式子用于看趋势，不应被当作线上可接受并发，因为真实请求是变长的，调度还需要为 prefill 和新请求保留空间；若扣除后预算小于零，应报告容量不可行，而不是输出负并发上限。

### 3.2.3 一个可复算的 8k 例子

假设 L=32、H_kv=8、D_h=128、T_ctx=8192、b=2，则一个请求的 cache 约为：

~~~math
M_{\mathrm{KV}}
=2\times32\times8192\times8\times128\times2
=1{,}073{,}741{,}824\text{ bytes}
\approx1024\text{ MiB}.
~~~

同样的模型，如果使用 MHA 且 H_kv=32，约为 4096 MiB；如果是 MQA 且 H_kv=1，约为 128 MiB。这里的比例来自 K/V head 数，不能据此断言 MQA 在所有任务上质量与 MHA 相同。

### 3.2.4 不能从参数量反推 cache

参数量主要决定权重规模和部分计算量，KV Cache 还取决于：

1. 层数和 K/V head 数。
2. head dimension 和 cache dtype。
3. 活跃请求的实际上下文总长度。
4. 是否保存全局历史、滑动窗口或压缩 latent。
5. tensor parallel 是否分片或复制 K/V。
6. block rounding、scale 和 metadata 的额外空间。

因此两个参数量相近的模型，长上下文并发能力可能差很多；一个采用 GQA 的模型也不一定比另一个采用 MLA 或混合 state 的模型拥有同样的 cache shape。部署前要读取模型 config、源码或模型卡中明确的 attention/cache 结构。

## 3.3 Prefill、decode 与位置状态

### 3.3.1 Prefill 如何写入 cache

输入有 T_in 个 token 时，prefill 会在每层产生整段 K/V，并把它们写入从位置 0 到 T_in-1 的状态槽。若使用 RoPE 或其他位置编码，写入 cache 的 K 通常已经包含相应的位置信息；后续 query 必须采用兼容的位置计算和 position ids。

这意味着 cache 不是可以脱离上下文随意搬运的普通 tensor。下列变化可能使一个看似相同的前缀不能复用：

1. 模型权重或 tokenizer revision 改变。
2. chat template、特殊 token 或 generation prompt 改变。
3. position ids、RoPE scaling、滑动窗口或 attention mask 配置改变。
4. cache dtype、量化 scale、layout 或并行拓扑不兼容。
5. 租户权限、工具 schema 或安全策略改变。

### 3.3.2 Decode 如何追加

如果输入 token 数是 T_in，已经生成了 t 个 token，下一次 decode 产生的新 K/V 对应位置 T_in+t。概念上可以写成：

~~~math
T_{\mathrm{ctx}}(t)=T_{\mathrm{in}}+t,
\qquad
K_l^{(t+1)}=K_l^{(t)}\mathbin{\|}K_{l,\mathrm{new}},
\qquad
V_l^{(t+1)}=V_l^{(t)}\mathbin{\|}V_{l,\mathrm{new}}.
~~~

符号 || 表示沿 sequence 维追加。实际 runtime 可能不真的执行一次大 tensor 的 concat，而是把新 token 写入下一个物理 block 的槽位；这是连续内存张量语义与分页存储实现之间的区别。

## 3.4 KV Cache 为什么加速，又为什么仍会变慢

### 3.4.1 被省掉的重复投影

设当前上下文长度为 T_ctx，hidden size 为 d。如果每一步都重新计算全部历史 token 的 K/V，历史投影的工作量数量级是：

~~~math
C_{\mathrm{no\ cache}}
\propto
T_{\mathrm{ctx}}d^2.
~~~

使用 cache 后，每一步只为一个新 token 计算 K/V：

~~~math
C_{\mathrm{with\ cache}}
\propto
d^2.
~~~

这说明为什么 cache 能避免大量重复计算。它并没有把整个 Transformer block 变成常数时间，因为新 token 仍需经过投影、MLP、归一化和 attention。

### 3.4.2 没有被省掉的历史读取

新 query 仍要和历史 key 做相似度，再聚合历史 value。attention 主项对单个 decode step 的数量级是：

~~~math
C_{\mathrm{decode,attn}}
\propto
H_qT_{\mathrm{ctx}}D_h.
~~~

因此“有 KV Cache”与“长上下文 decode 不变慢”是两个不同命题。cache 消除的是历史 K/V 的重复计算；它没有消除新 query 对这些历史状态的访问。

### 3.4.3 没有 cache 的复杂度直觉要谨慎

如果生成 N 个 token，没有 cache 时，历史 K/V 投影会反复覆盖从 T_in 到 T_in+N 的长度，累计工作大致呈二次增长；有 cache 时，历史投影变为一次 prefill 加上 N 次单 token 投影。attention 对历史的读取仍然随每一步上下文增长，因此总成本不会变成严格线性于输出长度的简单常数。

更准确的工程表述是：“KV Cache 将重复的历史 K/V 投影换成状态读取；它降低了大量重复计算，但 decode 的 attention 读取、模型权重访问和逐步调度仍然存在。”

## 3.5 MHA、MQA、GQA：K/V head 数的取舍

### 3.5.1 MHA 保存什么

Multi-Head Attention（MHA）中，query、key、value 通常都有同样数量的 head：

~~~math
H_{\mathrm{kv}}=H_q.
~~~

每个 query head 拥有独立的 K/V 表示，表达能力和实现兼容性较好，但 cache 体积最大。若模型有 32 个 query head、每个 head 128 维，K/V 需要存储 32 份 head。

### 3.5.2 MQA 如何缩小 cache

Multi-Query Attention（MQA）让多个 query head 共享同一组 K/V：

~~~math
H_{\mathrm{kv}}=1.
~~~

这样 cache 的 K/V head 因子从 H_q 降到 1，理论上把这部分显存和历史 K/V 读取降到 MHA 的 1/H_q。query 仍然可以有多个 head，模型计算并不是所有部分都减少；共享 K/V 还可能改变质量和训练稳定性。MQA 不能在服务配置里随意把一个已经训练好的 MHA 模型改成 MQA，除非模型架构、权重转换和质量回归支持这种改变。

### 3.5.3 GQA 是按组共享

Grouped-Query Attention（GQA）把 query head 分成若干组，每组共享一个 K/V head。若每个 K/V head 服务 g 个 query head，则：

~~~math
H_{\mathrm{kv}}=\frac{H_q}{g}.
~~~

在其他变量相同的情况下，GQA 相对 MHA 的 cache 比例是：

~~~math
R_{\mathrm{GQA/MHA}}
=\frac{M_{\mathrm{KV,GQA}}}{M_{\mathrm{KV,MHA}}}
=\frac{H_{\mathrm{kv}}}{H_q}
=\frac{1}{g}.
~~~

例如 H_q=32、H_kv=8 时，每 4 个 query head 共享一个 K/V head，cache 约为 MHA 的四分之一。这个比例只针对标准显式 K/V cache；MLA、滑动窗口和混合架构不能直接套用。

### 3.5.4 质量、带宽和并行之间的取舍

减少 K/V head 数通常带来三项收益：cache 变小、decode 读取字节减少、相同显存能容纳更多活跃上下文。但也有边界：

1. 共享 K/V 可能降低某些任务所需的表示自由度。
2. runtime 必须支持把 query head 正确映射到 K/V group。
3. tensor parallel 下 K/V head 是否可整除每个 rank 的分片数，会影响实现。
4. 仅减少 K/V 不会减少 MLP 和大部分权重读写。
5. 量化、kernel 和 GPU 拓扑可能决定实际收益是否达到理论比例。

因此部署比较应同时报告 cache bytes、TPOT、吞吐和独立质量结果，而不是只看一个显存百分比。

## 3.6 MLA、滑动窗口与递归 state：不都是同一种 cache

### 3.6.1 MLA 不能套用 H_kv 公式

Multi-head Latent Attention（MLA）类架构的思路不是简单减少 K/V head，而是把历史信息压缩到低维 latent，并可能额外保存与位置编码相关的解耦 key。用抽象符号表示，单层每个 token 的 cache 可以近似写成：

~~~math
M_{\mathrm{MLA/token/layer}}
\approx
(d_{\mathrm{latent}}+d_{\mathrm{pos}})b
+M_{\mathrm{scale/metadata}}.
~~~

这里 d_latent 是压缩的 KV latent 维度，d_pos 是位置相关状态维度；精确布局、是否保留可复用的投影量以及不同 kernel 的 workspace，要以具体实现为准。DeepSeek-V2 技术报告是 MLA 公开设计的主要一手资料之一。

工程上最重要的边界是：看到模型使用“multi-head”并不意味着可以用标准 MHA/GQA 的 2*L*H_kv*D_h*b 直接计算它的 cache。应从模型 config、实现代码或官方技术报告确认真正保存的是 full K/V、latent、位置 key、递归 state，还是几种状态的组合。

### 3.6.2 滑动窗口让部分 cache 随窗口而不是随全历史增长

如果某一 attention 层只允许查看最近 W 个 token，它的有效历史长度是：

~~~math
T_{\mathrm{effective}}=\min(T_{\mathrm{ctx}},W).
~~~

这里要求 `T_ctx\geq0` 且窗口 `W>0`。该层的 cache 预算可以按 `T_effective` 估算，而拥有 global attention 的层仍可能保留更长历史。混合模型不能用一个统一窗口替代所有层的状态。实现上还要考虑 RoPE、位置偏移、跨窗口摘要以及从 cache 中删除旧 token 后 attention mask 是否仍然正确。

### 3.6.3 递归或线性 attention 的 state

状态空间模型、线性 attention 或 gated delta rule 一类结构，可能用每个序列一个递归 state 汇总历史，而不是保存每个历史 token 的 full K/V。若模型包含 L_a 个显式 attention 层和 L_r 个递归层，可用下面的抽象估算：

~~~math
M_{\mathrm{history}}
\approx
B\sum_{l\in A}
2\min(T_{\mathrm{ctx}},W_l)H_{\mathrm{kv},l}D_{h,l}b_l
+B\sum_{l\in R}S_l b_l
+M_{\mathrm{metadata}}.
~~~

A 是显式 attention 层集合，W_l 是该层的窗口（global 层可以取很大的值），R 是递归层集合，S_l 是每个序列该层 state 的元素数。第二项对递归 state 不随 token 长度线性增长，但 state 可能是矩阵而不只是一个向量，因此必须使用实际元素数。

这类模型的部署风险从“cache token 数是否够”扩展为“每一步的 state transition 是否正确”：

1. batch 合并和拆分时，state 必须跟随 request id，而不是跟随 batch 下标。
2. request reset 必须清除旧 state，否则新请求可能继承旧上下文。
3. prefix sharing 对递归 state 不一定等价于复制 attention K/V。
4. state 的 dtype、量化和跨卡传输需要单独测量。

面对新的模型架构，先画出每一层的 history state layout，再选择 runtime；不要只根据参数量、模型名称或“支持超长上下文”的宣传语推断显存。

## 3.7 显存账本：cache 只是其中一项

### 3.7.1 总显存组成

对一张 GPU，可以把推理显存写成一个证据账本：

~~~math
M_{\mathrm{GPU}}
=M_{\mathrm{weights}}
+M_{\mathrm{KV}}
+M_{\mathrm{activation}}
+M_{\mathrm{workspace}}
+M_{\mathrm{communication}}
+M_{\mathrm{metadata}}
+M_{\mathrm{fragmentation}}
+M_{\mathrm{runtime}}.
~~~

M_weights 是权重和常驻模型 buffer，M_KV 是活跃请求状态，M_activation 是当前 prefill/decode 的中间激活，M_workspace 是 kernel 临时空间，M_communication 是 TP/PP/NCCL 等通信 buffer，其余项包含 block table、scale、allocator 和 runtime 自身开销。

如果 GPU 总容量为 M_total，可供动态 cache 使用的预算不能简单写成 M_total-M_weights，更合理的形式是：

~~~math
M_{\mathrm{KV,budget}}
=M_{\mathrm{total}}
-M_{\mathrm{weights}}
-M_{\mathrm{activation}}
-M_{\mathrm{workspace}}
-M_{\mathrm{communication}}
-M_{\mathrm{headroom}}.
~~~

M_headroom 用于吸收长度突发、临时分配、编译 kernel、驱动和监控误差。把所有剩余显存都分给 KV Cache，通常会把偶发 prefill 或长请求变成 OOM。

### 3.7.2 统一最大长度为什么会浪费

假设三个请求的当前长度是 680、3360 和 1420，若 allocator 为它们分别预留 1024、4096 和 2048 个 token 槽位，则实际预留 token 数是 7168；如果按真实长度计算，只有 5460 个 token。静态预留的额外槽位为：

~~~math
W_{\mathrm{reserved}}
=7168-5460=1708\text{ token slots}.
~~~

这些槽位不是模型需要的历史状态，而是为了避免动态扩容而提前占用的空间。变长请求越多、最大长度和平均长度差距越大，浪费越明显。分页分配把浪费限制到每个 block 的尾部，但会增加 block table 和访问间接层。

### 3.7.3 每个 rank 的账不能重复计算

在 tensor parallel 服务中，权重、K/V head 和通信 buffer 可能按 rank 分片，也可能为了 kernel 或通信效率复制一部分。若 r 个 rank 均匀分片 K/V head，某 rank 的理想 cache 规模约为：

~~~math
M_{\mathrm{KV,rank}}
\approx
\frac{2L H_{\mathrm{kv}}D_h b\,S_{\mathrm{ctx}}}{r}.
~~~

但只有在 head 可分片、layout 和实现确实如此时才能除以 r。一些架构的 K/V、latent 或通信状态会复制；pipeline parallel 则按层分配，不是把所有 cache 简单平均。容量报告必须注明“单卡显存”还是“整机总显存”，并给出每 rank 的实际读数。

## 3.8 PagedAttention：逻辑连续，物理分散

### 3.8.1 连续分配的困难

最朴素的 cache allocator 为每条请求预留一段连续空间。请求刚开始时不知道最终会生成多少 token，于是有三种选择：

1. 按最大上下文一次性预留，简单但浪费大。
2. 不断 realloc 和搬移，可能产生复制和同步开销。
3. 使用多个可增长片段，但 attention kernel 需要理解复杂布局。

在线服务中请求随时到达和结束，长度又高度不均匀。即使总空闲显存足够，也可能因为没有足够大的连续区域而分配失败；这就是外部碎片问题。已经预留但尚未使用的尾部空间则属于内部浪费。

### 3.8.2 Block table

PagedAttention 把逻辑序列切成固定大小的 token block，由 block table 把逻辑 block 映射到物理 block：

~~~text
logical sequence of request A: [0, 1, 2, 3]
physical block table:            [8, 2, 11, 4]

logical sequence of request B: [0, 1]
physical block table:            [5, 9]
~~~

attention kernel 仍然按逻辑顺序读取 token，只是在访问时通过 block table 找到物理位置。物理上不连续并不改变模型看到的 token 顺序；它改变的是 allocator 如何给状态找位置。

若 block size 为 S_block，第 i 个请求当前长度为 T_i，需要的 block 数为：

~~~math
N_i=\left\lceil\frac{T_i}{S_{\mathrm{block}}}\right\rceil.
~~~

该式要求 `T_i\geq0` 为整数且 `S_block>0` 为正整数；空序列需要零个 block，但不能使用零 block size 进行所谓的“自动计算”。

所有请求分配的 token 槽位和尾部浪费为：

~~~math
A_{\mathrm{paged}}
=S_{\mathrm{block}}\sum_{i=1}^{B}N_i,\qquad
W_{\mathrm{tail}}
=A_{\mathrm{paged}}-\sum_{i=1}^{B}T_i.
~~~

W_tail 是 token 槽位口径的内部浪费；还要加 block table、对齐、scale 和 allocator 元数据。block 越小，尾部浪费通常越小，但 table 更大、映射和 kernel 调度开销可能更高；block 越大，元数据更少，却可能浪费更多尾部空间。不存在脱离 workload 的万能 block size。

### 3.8.3 地址换算与 copy-on-write 轨迹

block table 不是抽象的“页号列表”，attention kernel 必须用它把逻辑 token 位置换成物理槽位。用教学 block size `S_block=4`，某请求的 table 是 `[8, 2, 11]`。逻辑 token 下标从 0 开始时：

~~~math
b_{\mathrm{logical}}=\left\lfloor\frac{i}{S_{\mathrm{block}}}\right\rfloor,\qquad
o=i\bmod S_{\mathrm{block}},\qquad
p=\mathrm{block\_table}[b_{\mathrm{logical}}],\qquad
\mathrm{slot}=pS_{\mathrm{block}}+o.
~~~

例如 `i=9` 落在逻辑 block 2 的 offset 1；table[2] 是物理 block 11，简化的一维 token slot 是 `11×4+1=45`。真实 cache 的 slot mapping 还要结合层、K/V、head 和 kernel layout；这个乘加只解释地址翻译，不是通用 runtime ABI。

再看两个 decode 分支共享一个尚未填满的尾 block。下表假设分叉前已有 `p0…p5`，两条分支先共享物理 block 8 和 2：

| 时刻 | 分支 A 的 block table | 分支 B 的 block table | 物理 block 2 的引用数 | 操作 |
| --- | --- | --- | ---: | --- |
| 分叉前 | `[8, 2]` | `[8, 2]` | 2 | block 8 满 4 token；block 2 保存 `p4,p5` |
| A 追加 `a6` | `[8, 11]` | `[8, 2]` | 1 | 因 block 2 仍被 B 使用，A 分配 block 11，复制 `p4,p5` 后再写 `a6` |
| B 追加 `b6` | `[8, 11]` | `[8, 2]` | 1 | B 在仍属于自己的 block 2 中写 `b6` |

copy-on-write 的关键不是“复制整条 cache”，而是只隔离将被改写的共享尾 block；完整、只读的前缀 block 继续共享。若不先隔离，A 写入 offset 2 会覆盖 B 同一位置，两个候选的历史就被混在一起。引用数降到零的物理 block 才能回到 free pool。不同 runtime 可能通过禁止共享不完整 block、复制尾块或采用其他状态策略实现同一不变量，不能仅凭“支持 prefix cache”推定它的分叉语义；[PagedAttention 原论文的多候选生成部分](https://arxiv.org/abs/2309.06180)也将 block 引用计数和写时复制用于共享前缀。

### 3.8.4 前缀共享与 copy-on-write

若多条请求的前缀 token 完全一致，物理 block 可以由多个逻辑序列共享。一个常见实现会对共享 block 维护引用计数；当某条序列在共享前缀之后产生分叉时，新写入的 block 使用 copy-on-write 或新 block，不能直接覆盖仍被其他请求使用的内容。

共享 cache 的 key 至少应包含模型权重 revision、tokenizer/template 版本、位置配置、精度/layout、租户或权限域，以及 prefix 的 token id 哈希。只按字符串哈希而不绑定这些条件，可能得到数值不一致或跨租户泄漏。prefix cache 的命中还应按复用 token 数统计：一次命中 32 token 与一次命中 32k token 的收益不能只算成两个命中请求。

### 3.8.5 PagedAttention 不解决哪些问题

分页管理不能做到：

1. 让 KV Cache 的理论 token 成本变成零。
2. 消除新 query 对历史 K/V 的读取。
3. 自动扩大 GPU 总显存。
4. 让不兼容的模型结构共享同一 cache。
5. 让超出显存预算的请求无代价地继续运行。

它主要减少连续分配和动态长度带来的碎片，改善 block 的按需分配、回收和共享。吞吐是否提升，还取决于 kernel、block size、batch、cache 访问和调度。

### 3.8.6 PagedAttention 与 vAttention：分页由谁来做

这两条路线都希望让变长请求按需占用物理显存，分歧在于“地址映射放在哪一层”。PagedAttention 在 runtime 中维护逻辑 block 到物理 block 的 table，attention kernel 按 table 访问不连续的 KV；这让 block 分配、请求分叉、引用计数和 prefix 复用都可以由 serving 系统显式控制。vAttention 则预留连续的虚拟地址范围，等请求增长时再用 CUDA 虚拟内存 API 映射物理页，尽量沿用假设 cache 连续的 attention kernel。

| 维度 | PagedAttention | vAttention |
| --- | --- | --- |
| 解决的约束 | 连续预留让最大长度和变长请求浪费大量显存 | 既要按需映射物理显存，又希望 KV 在虚拟地址上连续 |
| 地址管理 | serving runtime 维护 block table；kernel 按逻辑块查物理块 | runtime 申请虚拟连续区间，CUDA VMM 按需映射物理页 |
| 直接收益 | 显式 block 生命周期、细粒度分配、共享和回收 | 能复用部分连续布局 kernel，减少在 kernel 内处理 block table 的需要 |
| 代价与限制 | kernel 要支持分页布局；table 查找、映射、调度和 block size 有成本 | 依赖 CUDA VMM 的粒度与映射行为；页粒度、分配延迟和回收时机仍需优化 |
| 更适合评估的问题 | 现有 runtime 是否围绕 block pool、prefix sharing 和多请求调度构建 | 目标 CUDA 栈能否提供所需 VMM 能力，连续 kernel 复用是否抵消映射成本 |

[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)报告 vLLM 在其模型和 workload 上、保持相近延迟时相对 FasterTransformer/Orca 的吞吐提升为 2–4 倍；[vAttention 原论文](https://arxiv.org/abs/2405.04437)在 Yi-6B、Llama-3-8B、Yi-34B 和 1–2 张 A100 的评估中，报告原版 FlashAttention kernel 路线相对 vLLM 最高 1.97 倍吞吐，并报告相对 paged FlashInfer kernel 最高 1.45 倍 TTFT 加速。两项 vAttention 数据本身也用了不同的 kernel 对照，三组 baseline、硬件和 workload 不能拼成直接排名，也不能推导任意新硬件上的胜负。vAttention 论文还专门处理 CUDA 页粒度和分配调用延迟；所以“借操作系统虚拟内存就没有显存管理问题”也不是成立的结论。选择时应测同一模型、同一长度分布、同一延迟目标下的 TTFT、TPOT、吞吐、KV 浪费、CPU 调度开销和失败恢复路径。

## 3.9 KV Cache 量化：减少字节不等于没有代价

### 3.9.1 理想空间收益

如果 cache 的数据部分从 b_old 字节变成 b_new 字节，在相同 token 数、shape 和无额外元数据的理想条件下：

~~~math
\frac{M_{\mathrm{new}}}{M_{\mathrm{old}}}
\approx
\frac{b_{\mathrm{new}}}{b_{\mathrm{old}}}.
~~~

这里要求 `M_old>0`、`b_old>0` 和 `b_new>0`，且两边的 token 数、shape 和状态范围确实相同。例如 FP16/BF16 到 INT8，数据部分理论上约减半；到 FP8 也常是 1 byte 数据部分。但 scale、zero point、对齐和 kernel workspace 会让实际比例偏离，某些 runtime 还会在计算前把 cache 转回更高精度。

### 3.9.2 量化和反量化

一种常见的对称或仿射量化可以写成：

~~~math
q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right)+z,\ q_{\min},q_{\max}\right),\qquad
\hat{x}=s(q-z).
~~~

x 是原始 K/V 值，q 是低精度整数，`s>0` 是 scale，z 是 zero point，q_min 和 q_max 是满足 `q_min <= q_max` 的整数表示范围，x_hat 是反量化近似。对称量化常取 z=0；scale 可以按 tensor、head、channel、token 或 block 统计。

粒度越细，通常越能跟踪局部分布，metadata 和读取 scale 的开销也越大；粒度越粗，管理简单，但异常值可能占据动态范围，使大部分值量化误差变大。K 和 V 的误差影响也不完全相同：K 参与 attention score，误差会改变权重分布；V 参与加权聚合，误差会改变输出内容。

### 3.9.3 KIVI：K 与 V 为什么沿不同维度量化

“把所有 cache 元素都转成 2 bit”会忽略 K、V 在 attention 中的作用不同。K 进入 `QK^T`，量化误差会改变 softmax 前后的注意力分布；V 进入按注意力权重加权的求和，误差沿着被选中 token 的 value 影响输出。因此，KIVI 先观察 cache 分布和 attention 输出误差，再分别选择量化轴：在其研究的模型中，K 的固定 channel 有较大 outlier，按 channel 分组可把 outlier 的动态范围限制在对应 channel；V 没有同样明显的 channel outlier，按 token 量化则把误差限制在单个 token，避免少数重要 token 被其他 token 拉宽量化范围。[KIVI 原论文](https://arxiv.org/abs/2402.02750) 的分析是在指定模型、层和任务上得到，不是对所有 Transformer 的统计定律。

KV 是逐 token 到达的流式状态，按 channel 量化 K 还带来实现问题：同一 channel 的一组 token 没收齐之前，scale 和 zero point 不稳定。KIVI 的状态因此分成“已量化组”和高精度 residual。用缩小的教学参数 `G=2`、`R=4` 追踪 6 个 token：K 的 token 1–2 已形成一组并量化，token 3–6 暂留高精度；V 的 token 1–2 已按 token 量化，最近的 token 3–6 暂留高精度。第 7 个 token 到来时，K 先把已满的 4-token residual 按 G 分组量化，再将 token 7 放入新的高精度 residual；V 则将 token 7 入队，把最旧的 token 3 逐 token 量化，继续保留最新 4 个 token。这里的 G、R 仅为看懂状态变化的玩具数值；论文实验使用的 group size 为 32、residual length 为 128。

| 时刻 | 已量化 K | 高精度 K 残留 | 已量化 V | 高精度 V 残留 |
| --- | --- | --- | --- | --- |
| 6 个 token 后，教学 `G=2,R=4` | token 1–2，按 channel 分组 | token 3–6 | token 1–2，按 token 量化 | token 3–6 |
| 再到来 token 7 | token 1–6，按 channel 分组 | token 7 | token 1–3 | token 4–7 |

这条路径要实现分组元数据管理、非对齐边界处理，以及和矩阵乘法融合的反量化 kernel；压缩比例不只是 `2/16`。还要加 scale/zero point、残留区、对齐、临时 workspace，并确认读取时的 kernel 是否直接消费低比特数据。KIVI 的系统实验以 Llama-2-7B、单张 A100 80GB 和 ShareGPT 合成 workload 为例，输入/输出平均长度约 161/338 token，并逐渐增大 batch 到 OOM；[论文报告](https://arxiv.org/abs/2402.02750)在相近峰值显存下 batch size 最多增至 4 倍、吞吐提高 2.35–3.47 倍。这里的吞吐收益包含低比特 cache 腾出容量后扩大 batch 的系统效应，不能当成单请求 kernel 提速，更不能把 2-bit 数据宽度直接解释为所有目标模型 cache 可缩小到八分之一或性能必然更快。

当前 runtime 的量化合同还可能与 KIVI 算法不同。例如 vLLM v0.25.0 文档描述 FP8 KV cache 的 per-tensor 和 per-attention-head scale 路径，并说明 head-wise 路径需要特定 backend 和校准流程；这类文档能说明该版本公开支持什么，不等于实现了 KIVI 的 2-bit、K per-channel/V per-token 方案。[vLLM Quantized KV Cache 文档](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/) 应按实际部署版本重新确认。

### 3.9.4 量化评估不能只看显存

至少要在以下维度比较 FP16/BF16 cache 与量化 cache：

| 维度 | 要测的内容 |
| --- | --- |
| 内存 | cache bytes、scale/metadata、峰值和碎片 |
| 性能 | TTFT、TPOT/ITL、长上下文读取、kernel 时间 |
| 质量 | 事实、代码、数学、长文档和多轮对话 |
| 稳定性 | 不同长度、batch、温度和并发下的误差 |
| 工程 | 硬件支持、转换流程、checkpoint/runtime 兼容 |

短 prompt 的质量回归通过，不代表长上下文 cache 量化没有误差；单请求跑得快，也不代表高并发下反量化 kernel 没有成为瓶颈。KIVI 等论文提供了 KV 量化方法和实验依据，vLLM 等 runtime 文档提供具体实现参数，二者都不能替代目标模型的独立回放。

## 3.10 Prefix Cache：复用什么，隔离什么

### 3.10.1 哪些前缀适合复用

常见共享前缀包括固定 system prompt、企业助手规则、稳定的工具 schema、RAG 模板和多轮对话中未改变的历史部分。若前缀 token 数为 T_shared，完整 prompt 为 T_prompt，最粗的 prefill 复用比例直觉是：

~~~math
R_{\mathrm{prefix}}
\approx
\frac{T_{\mathrm{shared}}}{T_{\mathrm{prompt}}}.
~~~

这个比例要求 `T_prompt>0` 且 `0\leq T_shared\leq T_prompt`。它不是实际延迟收益。前缀命中后仍可能需要读取 cache、做新 suffix 的 prefill、执行后续 decode；如果原先 prefill 主要受 MLP 而非 attention 限制，复用比例和延迟比例也不会完全相同。

### 3.10.2 Cache key 的组成

一个可复用 prefix 至少要验证：

1. 模型权重和 adapter revision。
2. tokenizer、chat template 和特殊 token 配置。
3. position ids、RoPE scaling、滑动窗口和 attention mask 语义。
4. cache dtype、量化 scale、layout 和并行拓扑。
5. 租户、用户、数据权限和敏感内容策略。
6. 工具 schema、系统规则和前缀的 token id 哈希。

同样的自然语言文本，如果模板多了一个 generation marker，最终 token id 就可能改变；同样的 token 前缀，如果来自不同租户，也不应在没有明确隔离策略时共享物理页。缓存命中率应同时报告 request hit rate、reused token rate 和实际节省的 prefill 时间。

这里要区分“业务上完整的复用命名空间”和“某个 runtime 实际参与哈希的字段”。vLLM 当前设计文档描述的 v1 做法是：每个 KV block 的 hash 同时依赖块内 token 和前序 block 的 hash；多模态输入还要把图像等 frontend hash 纳入 extra hash。文档也说明可用每请求 `cache_salt` 把不同信任域隔离到不同 hash 链，降低通过命中延迟推测他人前缀的风险。[vLLM Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)；[NDSS 2025 的 prompt leakage 研究](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/) 在特定多租户框架上展示了 cache sharing 的时序侧信道。salt 是 cache reuse namespace 的隔离措施，不是数据加密、删除证明或所有侧信道的完整安全边界；部署还需让盐值由受信任的租户/会话策略产生，并检查请求鉴权、日志和缓存生命周期。

### 3.10.3 从首个请求到缓存淘汰的状态轨迹

继续用 block size 为 4 的教学例子，假设请求 A 的 prompt token 序列是 `[a,b,c,d,e,f,g,h]`，请求 B 是 `[a,b,c,d,e,f,g,h,x,y]`，且它们属于同一模型、模板、位置配置和信任域：

| 时刻 | 发生的操作 | A 的两个前缀 block | B 的命中/计算 | block 的可淘汰状态 |
| --- | --- | --- | --- | --- |
| A prefill | 计算前 8 个 token；每个完整 block 生成带 parent hash 的 key | 两块 active ref=1 | 尚未到达 | active block 不可淘汰 |
| A 完成 | 释放 A 的 active 引用；保留策略允许缓存时将完整块留在 reusable pool | 两块 active ref=0、仍可查找 | 尚未到达 | reusable block 可在压力下被淘汰 |
| B 到达 | 用 token 和 prefix 链查找连续 full-block 命中 | 仍可复用 | 命中 8 token，两个块 active ref 增至 1；只需 prefill `x,y` suffix | 因已被 B 使用而不能淘汰 |
| B 完成并遇到内存压力 | 释放 B 的引用，调度器需要新 block 时运行淘汰策略 | 可恢复为 reusable ref=0 | 没有 active 请求继续引用 | 按策略失效 hash 并归还物理 block |

这个轨迹说明 prefix cache 是“有引用生命周期的已计算状态”，不是只查一个哈希就自动省掉整条请求。命中 full blocks 后，scheduler 仍要为 suffix 做 prefill，decode 仍逐 token 增长；按请求计的 hit 也不能代替 reused-token 数。若 B 的 token 在第一个 block 之后就分叉，不能把不匹配的后续 block 当作命中；若 cache salt 或其他 namespace 字段不同，即使 token 相同也应发生 miss。该轨迹刻意忽略调度并发、partial block、量化 scale 和多模态 payload，只展示 [vLLM v1 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/) 描述的 block hash、active ref 和 cached-free 淘汰关系。

### 3.10.4 失效和回收

权重、模板或安全策略变更后，旧 prefix cache 不能继续作为新 revision 的状态。进程重启、GPU 迁移、显存水位升高和 LRU 淘汰也会使命中率变化。活跃请求的 cache 一般不能像可复用前缀那样随意淘汰；如果必须 offload 或 recompute，要把额外延迟、CPU 内存和隐私风险纳入模型。

### 3.10.5 vLLM 与 TensorRT-LLM 的复用管理差异

相同的“复用 prefix”能力可以落成不同的数据结构和淘汰策略。vLLM v1 当前设计文档说明它用 parent hash + block token + extra hash 标识 block，只把 full blocks 放入缓存，并在 block 对象上维护引用数和 free queue；TensorRT-LLM 当前 KV Cache System 文档描述 radix tree 搜索、优先级 LRU，并可把 GPU 淘汰的状态复制到 secondary memory 保持可复用。后者文档也注明当前实现只淘汰 radix-tree leaf，对 limited-attention layers 并不理想。[TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)

| 设计选择 | vLLM v1 文档 | TensorRT-LLM 文档 | 读者应核验的影响 |
| --- | --- | --- | --- |
| 查找结构 | block hash 链，包含前序 block | radix search tree | 最长前缀查询和树/哈希维护成本不同 |
| 复用命名 | block token、parent hash、extra hash；支持多模态 hash 与 cache salt | 依赖 prompt/block reuse 配置与模型支持 | 模型、adapter、多模态输入和租户命名空间要怎样隔离 |
| 驻留策略 | cached-free block 进入 free queue，压力时按 LRU 失效 | prioritized LRU，可将 primary block offload 到 secondary memory | 热前缀保留、CPU/host memory 和数据传输成本 |
| 公开限制 | 只缓存 full blocks；版本/序列化算法影响跨环境确定性 | 当前文档说明只淘汰叶节点，有限窗口 attention 下有适配限制 | 结论必须绑定实际 runtime release、模型结构和配置 |

该对照只说明公开文档中的实现差异，并不是同一模型上的性能排名。上线前应把 reuse hit token、prefill 节省、GPU/host block 水位、淘汰、迁移字节、延迟和跨租户隔离用同一 trace 测量。

## 3.11 长上下文与 cache 淘汰策略

### 3.11.1 窗口、摘要和截断不是同一件事

当上下文超过预算时，系统可能采用不同策略：

| 策略 | 保留的状态 | 代价 |
| --- | --- | --- |
| hard truncation | 删除最旧 token | 可能丢失关键事实和指令 |
| sliding window | 每层保留最近窗口 | 全局依赖能力受限 |
| summarization | 用新摘要替代旧内容 | 摘要错误会成为永久事实 |
| retrieval refresh | 删除旧 cache，按需重新检索 | 增加检索和 prefill 延迟 |
| offload/recompute | 将状态移出 GPU 或重新计算 | 增加 PCIe/CPU/计算成本 |

它们都不是“把 cache 清空”这么简单。删除 token 后，position ids、attention mask、引用位置和工具状态必须保持一致；摘要也不能把权限、来源和未确认事实混成一段无来源文本。

### 3.11.2 Active cache 与 reusable cache

活跃请求 cache 是当前生成正确性所需的运行状态；可复用 prefix cache 是未来请求可能使用的性能优化。前者优先保证语义和请求隔离，后者可以按 LRU、成本收益和显存水位淘汰。把两者放在同一个“cache hit”指标里，会掩盖 OOM 发生在哪一层。

可以分别记录：

~~~math
M_{\mathrm{cache,total}}
=M_{\mathrm{active}}+M_{\mathrm{reusable}}+M_{\mathrm{metadata}}.
~~~

这里每一项都应是同一时间点、同一 GPU 或同一资源域内的非负字节数。若监控只报告一个总量，却没有说明 active、reusable 和 metadata 的观测边界，就无法决定是淘汰前缀还是减少活跃请求。当显存水位升高时，优先淘汰可重算的 reusable prefix；如果 active cache 也不足，系统应选择排队、拒绝、降级或有明确成本的 offload，而不是静默删除仍在生成请求的历史。

### 3.11.3 H₂O 与 StreamingLLM：有限预算保留哪些 token

当系统必须让 attention 只看有限数量的历史 token 时，“保留最近窗口”并不是唯一选择。H₂O 与 StreamingLLM 都以减少 KV 驻留量为目标，但它们保留历史的依据不同，不能笼统归为同一种滑动窗口。

| 方法 | 保留规则 | 要解决的问题 | 主要限制 |
| --- | --- | --- | --- |
| 最近窗口 | 只保留末尾 W 个 token | 状态简单、内存上限可预期 | 可能删除开头指令或中段证据；超出训练窗口后的行为可能退化 |
| H₂O | 保留累计 attention 分数较高的 heavy hitters，并留出近期 token 预算 | 在受限 KV 预算下避免只看近期文本、丢掉反复被关注的历史项 | 累计注意力是重要性代理，不是任务正确性证明；评测结果绑定论文的模型、数据和 cache budget |
| StreamingLLM | 保留开头 attention sinks 和末尾近期窗口 | 让按有限窗口训练的模型在流式输入中更稳定地持续生成 | 保留 sink 有助稳定性，不等于保留任意中间证据；不承诺长文档问答的随机检索能力 |

用 20-token 的教学序列 `t0…t19`、8 个 KV 槽演示选择差别：若 H₂O 的 toy policy 预算为 4 个 heavy hitters 加 4 个 recent tokens，且累计注意力排名把 `t3,t5,t8,t10` 选为 heavy hitters，则保留 `{t3,t5,t8,t10,t16,t17,t18,t19}`；若 StreamingLLM 预算为 4 个 sink 加 4 个 recent tokens，则保留 `{t0,t1,t2,t3,t16,t17,t18,t19}`。相同槽位数对应不同的历史保留假设。token 排名和 4+4 的预算只是说明机制的构造数据，不是任何论文推荐参数。

H₂O 的论文将累计 attention score 作为 heavy-hitter 代理，并与 recent KV 配合；StreamingLLM 论文根据 attention sink 现象保存初始 token 和近期窗口。二者都通过丢弃部分历史改变了可参与 attention 的状态，属于有损保留策略；这和把完整 KV 无损 offload 到 host，或把数值转成低精度，是不同的优化轴。[H₂O 原论文](https://arxiv.org/abs/2306.14048)、[StreamingLLM 原论文](https://arxiv.org/abs/2309.17453)。实验应覆盖中间位置检索、长文档 QA、指令跟随、代码/数学和长时间流式生成，并报告 retained tokens、任务质量、TPOT、吞吐和预算。若任务要求随时精确引用任意旧 token，有限槽位 eviction 不能只凭平均困惑度下降就判定可替代完整 cache。

系统侧还要在 eviction、quantization 和 offload 之间做明确选择：eviction 删除部分历史并改变 attention 可见集合；quantization 保留位置但近似数值；offload 保留完整数值、用传输或换页延迟换 GPU 容量；recompute 重新执行前向、用计算换状态。不同策略的质量、延迟和故障语义不能只通过一个“压缩率”比较。KV 管理综述可以用来发现方法谱系，但具体机制仍应回到原论文与 runtime 文档核实。[系统感知 KV cache 优化综述](https://arxiv.org/abs/2607.08057)。

### 3.11.4 1M context 的现实边界

接口接受 1M token、模型训练过长上下文、runtime 能分配 1M cache、模型能在中间位置找回证据，是四个不同命题。即使每个 token 的 cache 成本很小，单请求也可能占用大量显存；如果模型有多层 global attention，窗口策略也不会自动帮它保存全部能力。容量、有效检索、跨段推理和单位成功成本应分别测量，不能把 context window 的宣传上限当作可承载并发。

## 3.12 OOM 排查：从数字到账本

### 3.12.1 先确认 OOM 发生在哪个阶段

prefill OOM、decode 过程中 OOM 和模型加载 OOM 的原因不同：

| 阶段 | 常见直接压力 | 优先核对 |
| --- | --- | --- |
| load | 权重、量化转换、常驻 buffer | dtype、权重分片、workspace、显存余量 |
| prefill | 长 prompt 激活、attention workspace、cache 写入 | 输入长度、batch token、kernel 临时空间 |
| decode | cache 增长、block 分配、长输出 | active context、block 水位、完成释放、抢占 |
| 高并发突发 | 多条请求同时申请状态 | queue、admission、最大 batch、回收延迟 |

“权重能加载”只说明 M_weights 放得下，不能说明 prefill 和 decode 的动态状态有空间。

### 3.12.2 一套可复现的检查顺序

先保存发生问题时的 workload 和 revision，再按以下顺序计算：

1. 统计 active request 数以及每条请求当前 T_ctx，不要只记录平均长度。
2. 用模型真实的 L、H_kv、D_h 和 cache dtype 计算理论 M_KV。
3. 加上 block rounding、scale、table、TP/PP buffer 和 allocator 读数。
4. 分开查看权重、activation、workspace、通信和 cache 的峰值时间。
5. 检查完成、取消、超时和异常路径是否释放 block/reference。
6. 检查是否存在 static max-length reserve、padding、长请求和 prefix cache 占用。
7. 在降低并发、缩短输入、降低输出、切换 GQA/量化或增大 block 预算后逐项复现。

每次只改变一项，才能知道是哪个变量使问题消失。若直接把最大上下文砍半，服务可能暂时不 OOM，却没有说明真正的预算缺口来自 cache、workspace 还是泄漏。

### 3.12.3 诊断数字的三个分母

cache 排查中常见三种分母：

1. 每个请求的 token 数，用来解释单请求状态。
2. 所有 active request 的总 token 数，用来解释动态状态。
3. GPU 可用 cache budget，用来解释是否会触发分配失败。

例如“cache 使用率 80%”必须说明是 M_active/M_KV_budget，还是 runtime pool 的已分配 block 数；如果把可复用 prefix、空闲但已分配 block 和活跃状态放在一个分子里，运维决策会失真。

## 3.13 一个零依赖的 KV Cache 内存审计 demo

下面的 demo 用三条不同长度请求比较 MHA、GQA 和 MQA 的理论 cache，模拟静态最大长度预留与分页 block 分配的差异，并估算 INT8 数据部分的理想空间变化。它不实现真实 attention，也不模拟 GPU allocator；输出中的 diagnostics 是对教学假设的解释，不是线上发布结论。

~~~python
import math
from copy import deepcopy


requests = [
    {"id": "chat", "prompt_tokens": 600, "generated_tokens": 80, "reserved_tokens": 1024},
    {"id": "rag_long", "prompt_tokens": 3200, "generated_tokens": 160, "reserved_tokens": 4096},
    {"id": "agent", "prompt_tokens": 1200, "generated_tokens": 220, "reserved_tokens": 2048},
]

layers = 32
query_heads = 32
head_dim = 128
bytes_bf16 = 2
block_size = 128
gpu_budget_gib = 9.3
weight_gib = 7.5
workspace_gib = 1.0
kv_head_options = {"MHA": 32, "GQA": 8, "MQA": 1}


class CacheContractError(ValueError):
    """A cache sizing input is outside its defined domain."""


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheContractError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheContractError(f"{name} must be a non-negative integer")
    return value


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CacheContractError(f"{name} must be finite and non-negative")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise CacheContractError(f"{name} must be finite and non-negative")
    return value


def validate_request(request):
    if not isinstance(request, dict):
        raise CacheContractError("request must be a mapping")
    if not isinstance(request.get("id"), str) or not request["id"].strip():
        raise CacheContractError("request id must be non-empty text")
    prompt = nonnegative_int(request.get("prompt_tokens"), "prompt_tokens")
    generated = nonnegative_int(request.get("generated_tokens"), "generated_tokens")
    reserved = nonnegative_int(request.get("reserved_tokens"), "reserved_tokens")
    if prompt + generated <= 0:
        raise CacheContractError("active token count must be positive")
    if reserved < prompt + generated:
        raise CacheContractError("reserved tokens cannot be below active tokens")
    return prompt + generated


def validate_batch(items, block_size_value):
    positive_int(block_size_value, "block_size")
    if not isinstance(items, list) or not items:
        raise CacheContractError("requests must be a non-empty list")
    for request in items:
        validate_request(request)
    ids = [request.get("id") for request in items]
    if len(ids) != len(set(ids)):
        raise CacheContractError("request ids must be unique")


def validate_configuration():
    for name, value in (("layers", layers), ("query_heads", query_heads), ("head_dim", head_dim), ("bytes_bf16", bytes_bf16)):
        positive_int(value, name)
    finite_nonnegative(gpu_budget_gib, "gpu_budget_gib")
    finite_nonnegative(weight_gib, "weight_gib")
    finite_nonnegative(workspace_gib, "workspace_gib")
    if not kv_head_options:
        raise CacheContractError("kv head options must be positive integers")
    for value in kv_head_options.values():
        positive_int(value, "kv heads")
    if any(query_heads % value != 0 for value in kv_head_options.values()):
        raise CacheContractError("each kv head count must divide query heads")


validate_configuration()
validate_batch(requests, block_size)


def kv_bytes(tokens, kv_heads, dtype_bytes=bytes_bf16):
    nonnegative_int(tokens, "cache tokens")
    positive_int(kv_heads, "kv heads")
    positive_int(dtype_bytes, "dtype bytes")
    return 2 * layers * tokens * kv_heads * head_dim * dtype_bytes


def gib(num_bytes):
    finite_nonnegative(num_bytes, "bytes")
    return num_bytes / 1024**3


active_tokens = sum(
    request["prompt_tokens"] + request["generated_tokens"]
    for request in requests
)
reserved_tokens = sum(request["reserved_tokens"] for request in requests)

memory = {
    name: round(gib(kv_bytes(active_tokens, kv_heads)), 3)
    for name, kv_heads in kv_head_options.items()
}
reserved_memory_gqa = round(
    gib(kv_bytes(reserved_tokens, kv_head_options["GQA"])), 3
)
active_memory_gqa = memory["GQA"]

page_rows = []
for request in requests:
    active = request["prompt_tokens"] + request["generated_tokens"]
    blocks = math.ceil(active / block_size)
    allocated = blocks * block_size
    page_rows.append(
        {
            "id": request["id"],
            "active": active,
            "blocks": blocks,
            "waste": allocated - active,
        }
    )

paged_tokens = sum(row["blocks"] * block_size for row in page_rows)
paged_memory_gqa = round(
    gib(kv_bytes(paged_tokens, kv_head_options["GQA"])), 3
)
last_block_waste_tokens = sum(row["waste"] for row in page_rows)
reserved_waste_tokens = reserved_tokens - active_tokens
fixed_memory_gib = weight_gib + workspace_gib
diagnostics = {
    "gqa_reduces_mha": memory["GQA"] < memory["MHA"],
    "mqa_reduces_gqa": memory["MQA"] < memory["GQA"],
    "paged_rounding_overhead_positive": paged_memory_gqa > active_memory_gqa,
    "paged_plan_fits_budget": fixed_memory_gib + paged_memory_gqa <= gpu_budget_gib,
    "static_reserve_exceeds_budget": fixed_memory_gib + reserved_memory_gqa > gpu_budget_gib,
}

print("active_tokens={}".format(active_tokens))
print("kv_memory_gib={}".format(memory))
print("gqa_vs_mha_ratio={:.2f}".format(memory["GQA"] / memory["MHA"]))
print("mqa_vs_mha_ratio={:.3f}".format(memory["MQA"] / memory["MHA"]))
print("page_rows={}".format(page_rows))
print("paged_tokens={}".format(paged_tokens))
print("last_block_waste_tokens={}".format(last_block_waste_tokens))
print("reserved_waste_tokens={}".format(reserved_waste_tokens))
print("reserved_memory_gqa_gib={}".format(reserved_memory_gqa))
print("paged_memory_gqa_gib={}".format(paged_memory_gqa))
print("int8_gqa_gib={}".format(round(active_memory_gqa / 2, 3)))
print("fixed_memory_gib={}".format(fixed_memory_gib))
print("diagnostics={}".format(diagnostics))


def expect_contract_error(label, edit, block_size_value=block_size):
    broken = deepcopy(requests)
    edit(broken)
    try:
        validate_batch(broken, block_size_value)
    except CacheContractError as error:
        print(f"{label}: {error}")
    else:
        raise AssertionError(f"{label} should have failed")


expect_contract_error("zero block size", lambda items: None, block_size_value=0)
expect_contract_error("reservation below active", lambda items: items[0].update(reserved_tokens=1))
expect_contract_error("boolean token count", lambda items: items[0].update(prompt_tokens=True))
expect_contract_error("non-mapping request", lambda items: items.__setitem__(0, "invalid request"))
expect_contract_error("duplicate request id", lambda items: items[1].update(id=items[0]["id"]))
expect_contract_error("empty batch", lambda items: items.clear())
~~~

按这组教学参数运行，关键输出为：

~~~text
active_tokens=5460
kv_memory_gib={'MHA': 2.666, 'GQA': 0.667, 'MQA': 0.083}
gqa_vs_mha_ratio=0.25
mqa_vs_mha_ratio=0.031
page_rows=[{'id': 'chat', 'active': 680, 'blocks': 6, 'waste': 88}, {'id': 'rag_long', 'active': 3360, 'blocks': 27, 'waste': 96}, {'id': 'agent', 'active': 1420, 'blocks': 12, 'waste': 116}]
paged_tokens=5760
last_block_waste_tokens=300
reserved_waste_tokens=1708
reserved_memory_gqa_gib=0.875
paged_memory_gqa_gib=0.703
int8_gqa_gib=0.334
fixed_memory_gib=8.5
diagnostics={'gqa_reduces_mha': True, 'mqa_reduces_gqa': True, 'paged_rounding_overhead_positive': True, 'paged_plan_fits_budget': True, 'static_reserve_exceeds_budget': True}
zero block size: block_size must be a positive integer
reservation below active: reserved tokens cannot be below active tokens
boolean token count: prompt_tokens must be a non-negative integer
non-mapping request: request must be a mapping
duplicate request id: request ids must be unique
empty batch: requests must be a non-empty list
~~~

这里有三个值得注意的细节：分页仍然产生 300 个 token 槽位的尾部浪费，但比静态最大长度预留少；GQA 把理论 cache 从 MHA 的 2.666 GiB 降到 0.667 GiB；在固定权重和 workspace 后，分页方案落在教学预算内，而静态预留超出预算。真实 allocator 还需要加 table、对齐、scale、通信和 headroom，所以不能把 `paged_plan_fits_budget=True` 直接当成真实 GPU 一定不 OOM。

## 3.14 常见误解与边界

### 3.14.1 KV Cache 让 attention 变成常数时间

不是。它让历史 K/V 的投影不必每步重算，但新 query 仍要访问历史上下文。只有在模型本身使用局部窗口、压缩 latent 或递归 state 时，历史访问模式才可能发生结构性变化；这也意味着不能再套用同一个 full-KV 公式。

### 3.14.2 剩余显存都可以给 KV Cache

不是。prefill 的激活、attention workspace、通信 buffer、allocator 元数据和突发请求都需要空间。生产服务还需要 headroom，避免在编译、采样、监控或长度突发时把 cache pool 推到不可恢复的 OOM。

### 3.14.3 PagedAttention 消除了碎片和浪费

不完全是。它减少连续预留造成的外部碎片，并把增长控制到 block 粒度，但最后一个 block 仍可能有内部浪费，block table 和映射访问也有成本。block size 需要根据长度分布和 kernel 实测选择。

### 3.14.4 GQA 可以在部署配置里打开

通常不行。GQA 是模型架构和训练权重的一部分，runtime 可以识别已有的 num_key_value_heads，但不能无损地把一个 MHA checkpoint 改成 GQA。任何转换都需要权重处理、兼容的 attention 实现和质量回归。

### 3.14.5 MLA 只是 H_kv 更小的 GQA

不是。MLA 通过 latent 和位置相关状态改变 cache 表示方式；它的显存和 kernel 需要使用自己的 layout 计算。用标准 2*L*H_kv*D_h*b 公式估算 MLA，可能得到方向完全错误的结论。

### 3.14.6 INT8 KV Cache 一定省一半显存且没有质量代价

不是。数据部分的理想比例接近一半，但 scale、metadata、对齐和临时转换会减少实际收益；K/V 的误差还可能改变 attention 分布和长上下文质量。量化必须在代表性长度、任务和并发下单独评估。

### 3.14.7 Prefix cache 命中就能跨用户共享

不是。共享必须服从模型版本、模板、位置配置和租户权限。可复用的公共规则与含有用户私密历史的 prefix 不是同一类 cache；即使 token 完全相同，也不能绕过访问控制。

## 3.15 一个长上下文多租户服务的 cache 设计

考虑一个企业分析服务。每个租户都有固定的 system policy 和工具 schema，用户请求会附带 4k--32k token 的文档，服务需要支持并发分析，同时不能让一个租户复用另一个租户的私有历史。目标不是把所有请求都塞进显存，而是在给定质量、延迟和隔离条件下决定哪些状态值得保留。

先建立单 token 账本。假设模型有 32 层、32 个 query heads、8 个 K/V heads、head dimension 为 128，cache 使用 BF16。一个上下文 token 的理论 cache 成本为：

~~~math
m_{\mathrm{token}}
=2\times32\times8\times128\times2
=131072\text{ bytes}.
~~~

因此，8,192 个 token 约需要 1 GiB 的 GQA cache；如果同样结构使用 MHA，K/V head 从 8 增加到 32，理论值约变为 4 GiB。这个比例首先用于判断架构和并发的方向，实际预算还要扣除权重、workspace、通信和 headroom。不能因为某一条请求的 cache 计算为 1 GiB，就把 1 GiB 之外的剩余显存全部分给它；新请求的 prefill 和多个活跃请求会同时申请状态。

再看变长请求和分页。某一时刻三个请求的上下文长度分别是 680、3,360 和 1,420 token，block size 为 128。分页分配需要的槽位是：

~~~math
N_{\mathrm{slots}}
=128\left(\left\lceil\frac{680}{128}\right\rceil
+\left\lceil\frac{3360}{128}\right\rceil
+\left\lceil\frac{1420}{128}\right\rceil\right)
=5760.
~~~

真实状态只有 5,460 token，因此尾部浪费是 300 个槽位。若使用静态预留 1,024、4,096 和 2,048 个槽位，则总预留为 7,168，浪费达到 1,708 个槽位。分页不是没有浪费，而是把浪费限制在 block 粒度，并允许完成的请求把物理 block 归还池中。block size 越小，尾部浪费往往越小，但 block table 和间接访问开销会上升；这个参数需要用真实长度分布和 kernel 测量。

然后区分两类 cache。活跃 cache 保存当前请求继续生成所必需的状态，不能因为显存水位升高而静默淘汰；可复用 prefix cache 只是性能优化，可以在确认可重算后按 LRU 或成本收益淘汰。固定 system policy 的 prefix 可能适合复用，但用户文档、权限结果、工具返回和个人历史不应默认进入跨用户共享池。一个安全的缓存键至少要绑定：

~~~text
model_revision
tokenizer_and_template_revision
position_and_attention_revision
cache_dtype_and_layout
tenant_and_permission_domain
tool_schema_revision
prefix_token_hash
~~~

如果两个租户的 system 文本碰巧相同，也不能仅凭 token hash 就允许共享；还要判断该 prefix 是否包含租户策略、检索结果或其他可推断信息。prefix 命中指标还要记录复用 token 数和节省的 prefill 时间，不能只记录命中请求数。

最后决定超预算时的动作。可以按优先级淘汰可复用 prefix、限制新请求进入、把长请求放入单独队列、降低最大输出或将可重算状态 offload；不能直接删除活跃请求的历史。每种降级都要在质量和用户契约中有对应语义：截断可能丢证据，摘要可能引入错误，offload 可能增加延迟，拒绝则要返回可重试的明确原因。对 1M context，也要分别测单请求 cache、并发容量、证据检索、跨段推理和单位成功成本；接口可接受的长度不是这些指标的替代物。

这个设计过程把四个层次分开了：公式给出状态数量级，分页决定动态分配，prefix cache 决定可复用计算，权限和生命周期决定哪些状态可以被复用或淘汰。部署评审真正需要的是这些层次之间的证据链，而不是把“KV Cache”“PagedAttention”和“长上下文”合并成一个笼统的加速结论。

## 3.16 资料与证据边界

以下资料按“原始研究、官方实现文档、当前项目测量”分层使用。论文给出方法和实验条件，官方文档描述接口与实现，当前集群压测才决定实际收益。

1. [Hugging Face Transformers: KV Cache](https://huggingface.co/docs/transformers/main/en/kv_cache)：核对 DynamicCache、cache position 和生成过程中的 cache 接口；不代表所有 runtime 使用同一数据布局。
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：PagedAttention 原始论文，说明分页式 KV 管理、共享和 serving 吞吐实验。
3. [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437)：比较用户态 block table 与 CUDA 虚拟内存映射路线；论文的 kernel、GPU 和 workload 结果不能与其他论文直接排名。
4. [vLLM Paged Attention Design](https://docs.vllm.ai/en/latest/design/paged_attention.html)：核对 vLLM 当前 paged attention 的设计文档；实现细节会随版本变化。
5. [vLLM Automatic Prefix Caching Design](https://docs.vllm.ai/en/latest/design/prefix_caching/) 与 [功能说明](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)：核对 block hash、引用、淘汰、多模态 extra hash 与 cache salt；命中率和安全策略仍需项目自测。
6. [vLLM Quantized KV Cache v0.25.0](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/)：核对该版本公开的 FP8 cache scale、校准和 backend 选项；不能据此推断支持 KIVI 的 2-bit 格式。
7. [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)：核对 block reuse、优先级淘汰、offload 和 KV event 的当前公开说明；支持项随 manager、模型和 release 变化。
8. [Multi-Query Attention](https://arxiv.org/abs/1911.02150)：MQA 原始论文，说明共享 K/V 的动机与质量/效率取舍。
9. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)：GQA 原始论文，说明从 MHA 到较少 K/V head 的方法和实验。
10. [DeepSeek-V2: A Strong Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)：MLA 公开设计的重要技术报告；MLA cache 不能直接套用标准 MHA/GQA 公式。
11. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：K per-channel、V per-token、residual window 和论文评测；具体质量与吞吐绑定模型、任务和 A100 配置。
12. [H₂O: Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048) 与 [StreamingLLM](https://arxiv.org/abs/2309.17453)：分别研究累计 attention heavy hitters + recent tokens，以及 attention sinks + recent window；两者解决的问题和保留假设不同。
13. [I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/)：多租户 KV sharing 的时序侧信道研究；攻击结果绑定论文中的系统和威胁模型。
14. [Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization](https://arxiv.org/abs/2607.08057)：用作执行/调度、内存层级、表示压缩和保留策略的研究地图；具体机制回到对应原论文核验。
15. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)：递归 state 取代 full attention history 的原始研究之一。
16. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)：Mamba-2 及 SSM/attention 关系的研究资料，用于理解非 full-KV 状态的边界。
17. [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：核对模板和特殊 token 对最终前缀 token 序列的影响。

对于新发布模型，资料可信度应按顺序判断：官方技术报告或模型卡是否明确写出 attention/state layout；公开代码是否与报告一致；目标 runtime 是否真正支持该 layout；最后才是当前硬件上的 cache bytes、TPOT 和质量实测。社区帖子或模型名称中的“超长上下文”“线性 attention”不能单独证明部署内存恒定。

## 3.17 小结

KV Cache 保存的是每层历史 attention 的 K/V 状态。它通过复用历史投影减少重复计算，但新 query 仍要读取历史状态，所以长上下文 decode 的带宽和延迟压力不会自动消失。

标准 full-KV 模型的基础账本是 2 * L * H_kv * D_h * b * S_ctx；S_ctx 应该是所有活跃请求真实上下文长度之和。MHA、MQA 和 GQA 主要通过 K/V head 数改变这笔账，MLA、滑动窗口和递归 state 则改变了状态的表示方式，不能机械套用同一个公式。

PagedAttention 解决的是动态 cache 的 block 分配、地址映射、回收和共享，仍然会有 block 尾部浪费和 metadata 成本。KV Cache 量化减少数据字节，却需要处理 scale、反量化、kernel 和质量误差。Prefix cache 可以减少重复 prefill，但必须绑定 token 序列、版本、位置配置和权限。

真正的 KV Cache 工程不是把显存公式背下来，而是能把 active state、reusable prefix、权重、workspace、通信和 headroom 分开记账，能从 OOM 和 P99 trace 追到具体分配与调度原因，并能用代表性任务验证性能收益没有换来质量或安全回归。
