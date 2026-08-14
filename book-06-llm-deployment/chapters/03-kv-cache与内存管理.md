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
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(
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

其中 b 是每个元素的字节数，例如 FP16 和 BF16 通常是 2，FP8 或 INT8 的数据部分通常是 1。这个式子没有计入 block table、scale、对齐、复制、通信 buffer 和 allocator 元数据，所以它是容量模型的基础，不是最终显存读数。

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

这个量特别适合做容量规划。若可供 KV Cache 使用的显存预算为 M_budget，并且假定所有请求上下文长度都是 T_ctx，粗略并发上限为：

~~~math
B_{\mathrm{rough}}
\approx
\left\lfloor
\frac{M_{\mathrm{budget}}}
{2L T_{\mathrm{ctx}}H_{\mathrm{kv}}D_h b}
\right\rfloor.
~~~

M_budget 必须是扣除权重、workspace、激活、通信和安全余量后的预算。这个式子用于看趋势，不应被当作线上可接受并发，因为真实请求是变长的，调度还需要为 prefill 和新请求保留空间。

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

面试或设计评审中，最准确的说法是：“KV Cache 将重复的历史 K/V 投影换成状态读取；它降低了大量重复计算，但 decode 的 attention 读取、模型权重访问和逐步调度仍然存在。”

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

该层的 cache 预算可以按 T_effective 估算，而拥有 global attention 的层仍可能保留更长历史。混合模型不能用一个统一窗口替代所有层的状态。实现上还要考虑 RoPE、位置偏移、跨窗口摘要以及从 cache 中删除旧 token 后 attention mask 是否仍然正确。

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

所有请求分配的 token 槽位和尾部浪费为：

~~~math
A_{\mathrm{paged}}
=S_{\mathrm{block}}\sum_{i=1}^{B}N_i,\qquad
W_{\mathrm{tail}}
=A_{\mathrm{paged}}-\sum_{i=1}^{B}T_i.
~~~

W_tail 是 token 槽位口径的内部浪费；还要加 block table、对齐、scale 和 allocator 元数据。block 越小，尾部浪费通常越小，但 table 更大、映射和 kernel 调度开销可能更高；block 越大，元数据更少，却可能浪费更多尾部空间。不存在脱离 workload 的万能 block size。

### 3.8.3 前缀共享与 copy-on-write

若多条请求的前缀 token 完全一致，物理 block 可以由多个逻辑序列共享。一个常见实现会对共享 block 维护引用计数；当某条序列在共享前缀之后产生分叉时，新写入的 block 使用 copy-on-write 或新 block，不能直接覆盖仍被其他请求使用的内容。

共享 cache 的 key 至少应包含模型权重 revision、tokenizer/template 版本、位置配置、精度/layout、租户或权限域，以及 prefix 的 token id 哈希。只按字符串哈希而不绑定这些条件，可能得到数值不一致或跨租户泄漏。prefix cache 的命中还应按复用 token 数统计：一次命中 32 token 与一次命中 32k token 的收益不能只算成两个命中请求。

### 3.8.4 PagedAttention 不解决哪些问题

分页管理不能做到：

1. 让 KV Cache 的理论 token 成本变成零。
2. 消除新 query 对历史 K/V 的读取。
3. 自动扩大 GPU 总显存。
4. 让不兼容的模型结构共享同一 cache。
5. 让超出显存预算的请求无代价地继续运行。

它主要减少连续分配和动态长度带来的碎片，改善 block 的按需分配、回收和共享。吞吐是否提升，还取决于 kernel、block size、batch、cache 访问和调度。

## 3.9 KV Cache 量化：减少字节不等于没有代价

### 3.9.1 理想空间收益

如果 cache 的数据部分从 b_old 字节变成 b_new 字节，在相同 token 数、shape 和无额外元数据的理想条件下：

~~~math
\frac{M_{\mathrm{new}}}{M_{\mathrm{old}}}
\approx
\frac{b_{\mathrm{new}}}{b_{\mathrm{old}}}.
~~~

例如 FP16/BF16 到 INT8，数据部分理论上约减半；到 FP8 也常是 1 byte 数据部分。但 scale、zero point、对齐和 kernel workspace 会让实际比例偏离，某些 runtime 还会在计算前把 cache 转回更高精度。

### 3.9.2 量化和反量化

一种常见的对称或仿射量化可以写成：

~~~math
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right)+z,\ q_{\min},q_{\max}\right),\qquad
\hat{x}=s(q-z).
~~~

x 是原始 K/V 值，q 是低精度整数，s 是 scale，z 是 zero point，q_min 和 q_max 是可表示范围，x_hat 是反量化近似。对称量化常取 z=0；scale 可以按 tensor、head、channel、token 或 block 统计。

粒度越细，通常越能跟踪局部分布，metadata 和读取 scale 的开销也越大；粒度越粗，管理简单，但异常值可能占据动态范围，使大部分值量化误差变大。K 和 V 的误差影响也不完全相同：K 参与 attention score，误差会改变权重分布；V 参与加权聚合，误差会改变输出内容。

### 3.9.3 量化评估不能只看显存

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

这个比例不是实际延迟收益。前缀命中后仍可能需要读取 cache、做新 suffix 的 prefill、执行后续 decode；如果原先 prefill 主要受 MLP 而非 attention 限制，复用比例和延迟比例也不会完全相同。

### 3.10.2 Cache key 的组成

一个可复用 prefix 至少要验证：

1. 模型权重和 adapter revision。
2. tokenizer、chat template 和特殊 token 配置。
3. position ids、RoPE scaling、滑动窗口和 attention mask 语义。
4. cache dtype、量化 scale、layout 和并行拓扑。
5. 租户、用户、数据权限和敏感内容策略。
6. 工具 schema、系统规则和前缀的 token id 哈希。

同样的自然语言文本，如果模板多了一个 generation marker，最终 token id 就可能改变；同样的 token 前缀，如果来自不同租户，也不应在没有明确隔离策略时共享物理页。缓存命中率应同时报告 request hit rate、reused token rate 和实际节省的 prefill 时间。

### 3.10.3 失效和回收

权重、模板或安全策略变更后，旧 prefix cache 不能继续作为新 revision 的状态。进程重启、GPU 迁移、显存水位升高和 LRU 淘汰也会使命中率变化。活跃请求的 cache 一般不能像可复用前缀那样随意淘汰；如果必须 offload 或 recompute，要把额外延迟、CPU 内存和隐私风险纳入模型。

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

当显存水位升高时，优先淘汰可重算的 reusable prefix；如果 active cache 也不足，系统应选择排队、拒绝、降级或有明确成本的 offload，而不是静默删除仍在生成请求的历史。

### 3.11.3 1M context 的现实边界

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


def kv_bytes(tokens, kv_heads, dtype_bytes=bytes_bf16):
    return 2 * layers * tokens * kv_heads * head_dim * dtype_bytes


def gib(num_bytes):
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
~~~

这里有三个值得注意的细节：分页仍然产生 300 个 token 槽位的尾部浪费，但比静态最大长度预留少；GQA 把理论 cache 从 MHA 的 2.666 GiB 降到 0.667 GiB；在固定权重和 workspace 后，分页方案落在教学预算内，而静态预留超出预算。真实 allocator 还需要加 table、对齐、scale、通信和 headroom，所以不能把 paged_plan_fits_budget=True 直接当成真实 GPU 一定不 OOM。

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

## 3.15 面试与设计评审中的完整回答

问“KV Cache 为什么能加速”时，先讲计算因果：自回归 decode 每步都需要历史 K/V；没有 cache 就会重复计算历史 token 的 K/V，有 cache 则只计算新 token 的 K/V，再让新 query 读取历史状态。然后补上代价：cache 占显存，attention 读取仍随上下文增长，所以长上下文服务仍可能 memory-bound。

问“如何估算显存”时，给出变量完整的式子：2 * L * H_kv * D_h * b * S_ctx，其中 S_ctx 是所有 active request 的真实上下文 token 总数。再说明需要额外加权重、activation、workspace、通信、block rounding、scale 和 headroom；如果是 TP，要核对 K/V 是分片还是复制，不能机械除以 GPU 数。

问“PagedAttention 解决什么问题”时，说明它是内存管理和地址映射方法：逻辑序列按固定 block 切分，由 block table 映射到不连续的物理 block，支持按需增长、回收和前缀共享。它不能消除 cache 的理论成本，也不能替代调度、量化或质量评估。

问“长上下文为什么难”时，完整回答应同时包含 prefill 的输入计算、每步 decode 的历史状态访问、cache 显存随 token 数增长、尾延迟和任务质量风险。应按 workload 测 TTFT、TPOT、P99、cache bytes、抢占/拒绝和质量，而不是只引用 context window 上限。

## 3.16 资料与证据边界

以下资料按“原始研究、官方实现文档、当前项目测量”分层使用。论文给出方法和实验条件，官方文档描述接口与实现，当前集群压测才决定实际收益。

1. [Hugging Face Transformers: KV Cache](https://huggingface.co/docs/transformers/main/en/kv_cache)：核对 DynamicCache、cache position 和生成过程中的 cache 接口；不代表所有 runtime 使用同一数据布局。
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：PagedAttention 原始论文，说明分页式 KV 管理、共享和 serving 吞吐实验。
3. [vLLM Paged Attention Design](https://docs.vllm.ai/en/latest/design/paged_attention.html)：核对 vLLM 当前 paged attention 的设计文档；实现细节会随版本变化。
4. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)：核对 prefix block 复用的官方功能说明；命中率和安全策略仍需项目自测。
5. [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)：核对具体量化 cache 的支持边界；硬件、模型和量化格式决定实际速度。
6. [TensorRT-LLM KV Cache Reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)：核对 NVIDIA runtime 的 cache reuse 说明和配置边界。
7. [Multi-Query Attention](https://arxiv.org/abs/1911.02150)：MQA 原始论文，说明共享 K/V 的动机与质量/效率取舍。
8. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)：GQA 原始论文，说明从 MHA 到较少 K/V head 的方法和实验。
9. [DeepSeek-V2: A Strong Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)：MLA 公开设计的重要技术报告；MLA cache 不能直接套用标准 MHA/GQA 公式。
10. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：KV 量化方法和实验依据；论文结果不等于所有模型和 runtime 的质量保证。
11. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)：递归 state 取代 full attention history 的原始研究之一。
12. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)：Mamba-2 及 SSM/attention 关系的研究资料，用于理解非 full-KV 状态的边界。
13. [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：核对模板和特殊 token 对最终前缀 token 序列的影响。

对于新发布模型，资料可信度应按顺序判断：官方技术报告或模型卡是否明确写出 attention/state layout；公开代码是否与报告一致；目标 runtime 是否真正支持该 layout；最后才是当前硬件上的 cache bytes、TPOT 和质量实测。社区帖子或模型名称中的“超长上下文”“线性 attention”不能单独证明部署内存恒定。

## 3.17 小结

KV Cache 保存的是每层历史 attention 的 K/V 状态。它通过复用历史投影减少重复计算，但新 query 仍要读取历史状态，所以长上下文 decode 的带宽和延迟压力不会自动消失。

标准 full-KV 模型的基础账本是 2 * L * H_kv * D_h * b * S_ctx；S_ctx 应该是所有活跃请求真实上下文长度之和。MHA、MQA 和 GQA 主要通过 K/V head 数改变这笔账，MLA、滑动窗口和递归 state 则改变了状态的表示方式，不能机械套用同一个公式。

PagedAttention 解决的是动态 cache 的 block 分配、地址映射、回收和共享，仍然会有 block 尾部浪费和 metadata 成本。KV Cache 量化减少数据字节，却需要处理 scale、反量化、kernel 和质量误差。Prefix cache 可以减少重复 prefill，但必须绑定 token 序列、版本、位置配置和权限。

真正的 KV Cache 工程不是把显存公式背下来，而是能把 active state、reusable prefix、权重、workspace、通信和 headroom 分开记账，能从 OOM 和 P99 trace 追到具体分配与调度原因，并能用代表性任务验证性能收益没有换来质量或安全回归。
