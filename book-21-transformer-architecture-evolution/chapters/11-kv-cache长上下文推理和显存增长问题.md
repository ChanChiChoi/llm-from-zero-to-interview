# 第十一章：KV Cache、长上下文推理和显存增长问题

上一章统计 attention 在一段序列上建立多少 query-key 交互。本章把视线移到自回归生成：模型每次只追加一个 token，哪些历史计算可以复用，哪些历史状态必须继续保留？

读完本章，读者应能沿一次生成过程说明 KV cache 如何更新；按模型层数、活跃请求长度、KV head 数和数据类型估算显式 KV 的容量；并区分模型结构改变了多少历史信息、serving 系统怎样摆放这些状态、目标硬件上实际需要测量什么。

这里以因果 Transformer 的常规全局 attention 为主。MQA/GQA/MLA 的架构细节由[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)展开，attention 计算由[第十章](10-o-n2-attention的计算与显存瓶颈.md)展开，稀疏模式、位置扩展和推理系统的完整实现分别见[第十九章](19-Sliding-Window与稀疏注意力.md)、[第二十章](20-FlashAttention与PagedAttention.md)及相关专题。本章用它们解释 KV 容量的边界，不重复完整实现教程。

## 11.1 从一次生成看历史状态怎样复用

设输入 prompt 有 $n$ 个 token。Prefill 阶段一次处理整个 prompt，并在每层保存这些位置产生的 key 和 value。prompt 最后一个位置的输出分布可用于采样第一个新 token。

要继续预测后续 token 时，模型把新 token 逐层送入网络。令 $h_t^{(\ell)}$ 为第 $\ell$ 层当前 token 的输入， $W_Q^{(\ell)}$、$W_K^{(\ell)}$、$W_V^{(\ell)}$ 是该层学习到的投影矩阵；这个位置计算新的 query、key 和 value：

~~~math
q_t^{(\ell)}=h_t^{(\ell)}W_Q^{(\ell)},\qquad
k_t^{(\ell)}=h_t^{(\ell)}W_K^{(\ell)},\qquad
v_t^{(\ell)}=h_t^{(\ell)}W_V^{(\ell)}
~~~

将新的 key 和 value 加入本层可见集合后，该位置的 query 读取历史位置和当前自己的 key，并用对应的 value 汇总结果。新 K/V 留在 cache 中，供之后的位置读取。每一层都有自己的历史状态；一层的输出还是下一层新位置计算的输入。

缓存之所以能复用，是因为在因果注意力中，未来 token 不会改写较早位置的隐藏状态。在参数、旧 token、位置编号和旧位置可见规则不变，且逐位置运算是确定的条件下，数学上缓存计算与每步重算整个前缀结果相同；不同实现的浮点舍入仍可能略有差异。[Stanford MS&E 319 的 KV caching 讲义](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf)给出了这个等价条件和逐层更新过程。若位置规则或 attention pattern 会随输入长度变化，需按具体模型核对等价条件。

KV cache 是推理时复用的运行状态，不是模型权重。关闭 cache 仍可生成文本，但每一步都会重新计算已有前缀，通常浪费大量工作。

## 11.2 Prefill 与 Decode 做的工作不同

| 阶段 | Query 与 Key 的长度 | 这一步主要做什么 |
| --- | --- | --- |
| Prefill | 对长度为 $n$ 的 prompt， $T_q=T_k=n$ | 并行处理 prompt 的各位置，并建立每层历史 K/V |
| Decode | 每步 $T_q=1$ ， $T_k$ 随已处理历史增长 | 计算新位置，并让它读取已有 K/V |

生成第一个 token 通常直接使用 prefill 末位置的 logits。此后每个新增位置贡献一行 query-key 交互。先假设批次里的 $B$ 条请求都有长度为 $n$ 的已缓存前缀，且各处理 $m$ 个新增位置； $H_q$ 表示 query head 数。第 $j$ 个新增位置可读取 $n+j$ 个 key，这些 decode 行合计包含：

~~~math
N_{\mathrm{decode\ pairs}}
= B H_q \sum_{j=1}^{m}(n+j)
= B H_q\left(mn+\frac{m(m+1)}{2}\right)
~~~

该式只数位置交互，不含每层投影、MLP、softmax 等其他运算。请求长度不同时，应按各请求的 query 行和历史长度分别求和。cache 避免了逐步重算旧位置的投影和前馈网络，但新 query 仍要读取随历史长度增长的 K/V。完整的 prefill 与 decode 复杂度口径见[第十章](10-o-n2-attention的计算与显存瓶颈.md)。

## 11.3 一条请求的 cache 里有多少数据

对一层全局 attention，令 $B$ 为请求数、$H_{\mathrm{kv}}$ 为 K/V head 数、$T$ 为每个请求已保留的 token 数、$d_h$ 为每个 head 的维度。常见 K/V 张量可以写成：

~~~math
K_{\mathrm{cache}},V_{\mathrm{cache}}
\in \mathbb{R}^{B\times H_{\mathrm{kv}}\times T\times d_h}
~~~

普通 MHA 中 $H_{\mathrm{kv}}=H_q$ ；MQA/GQA 可让多个 query heads 共用较少的 K/V heads。实际实现可能把维度换序、切成 blocks 或按并行设备分片，但这些变化本身不改变逻辑上需要保留的状态。

假设有 $L$ 层，每层使用相同的 K/V head 数和 head dimension，每个 cache 元素占 $s$ bytes，则显式全局 KV 的 payload 为：

~~~math
M_{\mathrm{KV}}=2LBTH_{\mathrm{kv}}d_hs\ \mathrm{bytes}
~~~

因子 $2$ 来自 K 和 V 两份。若第 $i$ 个活跃请求的已保留长度为 $T_i$ ，把 $BT$ 换成各请求长度之和：

~~~math
M_{\mathrm{KV}}=2LH_{\mathrm{kv}}d_hs\sum_{i=1}^{B}T_i\ \mathrm{bytes}
~~~

这是 cache 数据本身的估算。它不包含模型权重、attention workspace、block metadata、对齐和 allocator 预留，也没有判断这些数据会如何分到不同 GPU。对混合 attention、滑动窗口或递归状态模型，应逐层记录状态类型和大小，不能把同一个全局 attention 公式套到每层。

### 一个可以手算的容量例子

设 $L=32$、$B=16$、每个请求保留 $T=8192$ 个 token、$H_{\mathrm{kv}}=32$、$d_h=128$ ，并用 BF16 的 $s=2$ bytes 估算：

~~~math
M_{\mathrm{KV}}
=2\times32\times16\times8192\times32\times128\times2
=68{,}719{,}476{,}736\ \mathrm{bytes}
=64\ \mathrm{GiB}
~~~

这是 16 个相同长度活跃请求的 KV payload，不是整卡总显存。如果其他条件不变而 $H_{\mathrm{kv}}=8$ ，payload 为 $16$ GiB。对单个请求而言，每新增长一个 token 的 cache 约为：

~~~math
2LH_{\mathrm{kv}}d_hs
=131{,}072\ \mathrm{bytes}
=128\ \mathrm{KiB}
~~~

所以可以先算每请求、每 token 的成本，再乘当前活跃请求的实际长度。最大并发不能只用一个“batch size”代替：长度不同的请求占用也不同，分页尾槽、权重、workspace 和并行布局还要另外记账。

## 11.4 Decode 为什么会受历史 K/V 的搬运限制

每个 decode query 需要和可见历史 keys 计算分数，并从对应 values 汇总结果。历史越长，单步 attention 读取的状态越多；batch 越大，系统同时保留的请求状态也越多。Shazeer 提出 MQA 时，正是以增量解码重复加载大型 K/V 张量的内存带宽成本作为动机。[原论文](https://arxiv.org/abs/1911.02150)

这不是说任何 decode 请求都必然受显存带宽限制。短上下文、不同 batch、不同 head 结构、权重读取、kernel 和硬件缓存都会改变瓶颈。公式能估容量和历史状态的量级；实际 token latency 或吞吐需要绑定模型、输入长度分布、并发、设备、数据类型和推理实现测量。

Prefill 与 decode 的延迟口径也不同：prompt 变长会增加 prefill 计算和首 token 等待；长历史会增加后续 decode 每步要处理的状态。端到端首 token 延迟还会受排队和调度影响，不能只用 attention 复杂度估出。

## 11.5 哪些方法改变状态量，哪些只改变管理方式

这些方法作用在不同层。比较时先问它改了每 token 保存的表示、允许直接读取的历史，还是状态在设备上的分配方式。

| 方法 | 改变什么 | 仍需核对的边界 |
| --- | --- | --- |
| MHA、MQA、GQA | 显式 K/V head 数不同；在每 head 维度和精度相同的情况下，cache payload 随 $H_{\mathrm{kv}}$ 成比例变化 | MQA/GQA 的质量与实际延迟不能仅由 head 数比例推出；论文结论受模型、训练和硬件设置约束。定义与证据见[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md) |
| MLA | 将 K/V 相关表示压到较低维 latent；DeepSeek-V2 还把 content latent 与位置相关的 RoPE key 分开缓存 | 不能简单描述成“存 latent，需要时完整还原 K/V”；具体维度、投影吸收方式和实现见[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)及[DeepSeek-V2 原论文](https://arxiv.org/html/2405.04434v5) |
| Sliding-window attention | 每层只需直接保留窗口内可读取的 K/V，窗口长度限制该层的直接历史读取 | 多层计算可把较早信息间接传到当前表示；窗口限制不等于系统绝无较远历史信息。机制见[第十九章](19-Sliding-Window与稀疏注意力.md)和[课程讲义](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf) |
| 递归或线性状态 | 用模型定义的状态更新表示历史，不再等同于逐 token 保存完整 K/V | 状态形状、是否固定、可保留的信息及更新成本都依赖具体架构；见[KDA](61-kda递归注意力.md)与[递归状态 cache](71-recursive-state-cache递归状态缓存.md) |
| KV 量化或 offload | 改变状态表示精度或存储层级 | 字节数、反量化/传输开销、质量和延迟依赖具体方法与部署配置，需分开评估；本章不把压缩率或速度写成通用值 |

MLA 的 KV cache 压缩来自其特定 latent 和位置编码设计。DeepSeek-V2 的论文同时比较了整套模型与基线；其中端到端吞吐差异不能只归因给 MLA 这一项。[MQA 原论文](https://arxiv.org/abs/1911.02150)与 [GQA 论文](https://aclanthology.org/2023.emnlp-main.298/)也报告了特定模型和评估设置，不能据此给所有模型承诺固定质量或速度收益。

## 11.6 PagedAttention 改变物理分配，不改 cache 公式

请求长度在生成中逐步增长，也可能随完成而释放。若系统预先给每个请求保留最大长度的连续区域，会为尚未生成的 token 占住空间；请求长度不同时，也可能产生未使用槽位。

PagedAttention 把逻辑 K/V 序列划成固定 token 数的 blocks，再用 block table 把逻辑 blocks 映射到物理显存。新 token 到来时按需分配 block，逻辑相邻的 blocks 不必位于连续物理地址。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)展示了该设计如何减少最大长度预留造成的浪费，并支持共享相同前缀的 cache blocks。

分页不会让同一组活跃 token 的 K/V 数学 payload 自动变小。它改变分配、尾部空槽和复用方式；共享相同前缀时，物理存储可以避免重复。每个请求的尾块最多留出 block size 减一处 token 空槽。block size 也会影响 kernel 的处理方式和硬件利用率。具体引擎的实现与能力应按版本核对，完整系统机制见[第二十章](20-FlashAttention与PagedAttention.md)。

Continuous batching 让新请求在 decode 迭代之间加入、已完成请求退出；调度器由此改变当前活跃状态集合。它可以与分页 KV 管理配合，但它不减少每个活跃 token 本身需要的 cache payload。prefix cache 也只有在请求具有兼容的 token 前缀、位置和模型状态时才能复用；不同引擎的匹配条件并不相同。

## 11.7 长上下文能力不等于 cache 预算或有效检索

模型或服务声明能接收某个长度，只说明它在指定配置下接受相应长度的输入；这不表示该长度与目标并发一定能同时放入目标设备，也不保证模型能从任意位置可靠取出相关证据。容量估算关注保留多少状态；有效长上下文能力还需要任务与位置条件下的评测，见[第十三章](13-长距离依赖与lost-in-the-middle.md)。

FlashAttention 减少 attention 计算中的数据搬运和中间矩阵存储，但不自动删除自回归 decode 后续需要读取的历史 KV 状态。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)讨论的是 attention 的 IO-aware 计算；两类 cache/内存问题的区别和 kernel 细节见[第二十章](20-FlashAttention与PagedAttention.md)。

做一次容量评估时，可以按这条顺序：

1. 按层列出历史表示：全局 KV、有限窗口、latent 或递归状态。
2. 记录当前活跃请求及其实际保留长度，不用最大并发人数代替长度分布。
3. 用模型真实的 head 数、状态维度和精度计算逻辑 payload。
4. 再核对分页尾槽、前缀共享、offload、并行分片、workspace 和设备预留。
5. 若要判断吞吐、延迟或质量，另行固定 workload 与运行环境测量；容量公式不产生通用的上线阈值。

因此，KV cache 的增长由模型每层保存什么和系统同时保留多少历史共同决定。结构压缩、历史筛选、量化、分页和调度解决的是不同问题，不能把它们合并成一个“长上下文优化”指标。
