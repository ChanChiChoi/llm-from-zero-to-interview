# 第十章：O(n²) Attention 的计算与显存瓶颈

上一章讨论了 Transformer 的并行路径与 scaling 证据。本章把视线缩到一个具体成本：每个 query 需要和多少个 key 建立交互，以及这些交互怎样转成计算、存储和数据搬运。

读完本章，读者应能做三件事：从张量形状数出 attention 的 pair 和主要 FLOPs；分清平方计算量与“必须存一张平方矩阵”不是同一件事；判断训练、prefill、decode 和不同 attention 改造分别改变了什么。本章只用理论账本解释机制，不把复杂度当成实际墙钟时间，也不把某一框架或硬件的表现推广成通用结果。

## 10.1 从张量形状看 attention 在计算什么

Scaled dot-product attention 先用 query 与 key 的点积表示匹配程度，再按每个 query 的所有 key 归一化，并以这些权重汇总 value：

~~~math
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
~~~

$M$ 表示可选的可见性掩码；不可见的位置在 softmax 前被排除。缩放项 $\sqrt{d_h}$ 用来控制点积的尺度，标准 Transformer 论文在独立、零均值、单位方差的分量假设下说明了这一设计动机。具体定义见 [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)。

设 batch size 为 $B$，query head 数为 $H_q$，query 长度为 $T_q$，key/value 长度为 $T_k$，head dimension 为 $d_h$。常见张量形状为：

~~~math
Q\in\mathbb{R}^{B\times H_q\times T_q\times d_h},\qquad
K,V\in\mathbb{R}^{B\times H_{kv}\times T_k\times d_h}
~~~

~~~math
S,P\in\mathbb{R}^{B\times H_q\times T_q\times T_k},\qquad
O\in\mathbb{R}^{B\times H_q\times T_q\times d_h}
~~~

在普通多头注意力中，$H_{kv}=H_q$；GQA/MQA 的 K/V head 数可以更少，K/V 会按 query-to-KV head 映射供多个 query heads 使用。公式没有展开这个映射，但 score 张量仍按 $H_q$ 计。两类结构差异以及它们对 KV cache 的影响由[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)和[第十一章](11-kv-cache长上下文推理和显存增长问题.md)展开。

$S$ 中每个元素对应一对 query-key 位置。因此，dense attention 的 pair 数是：

~~~math
N_{\mathrm{pair}}=B H_q T_q T_k
~~~

对于 self-attention，$T_q=T_k=T$，于是 pair 数随 $T^2$ 增长。对于 cross-attention，两个序列长度可以不同，成本由 $T_qT_k$ 决定。注意这里说的是位置交互数量；它还没有说明实现是否把整张 $S$ 或 $P$ 存下来。

## 10.2 二次项分别带来多少计算与存储

忽略 projection、softmax、mask 和其他逐元素操作，只计算 $QK^\top$ 与 $PV$ 两个主要矩阵乘；按一次乘加记为 2 FLOPs：

~~~math
F_{QK}\approx 2B H_qT_qT_kd_h,\qquad
F_{PV}\approx 2B H_qT_qT_kd_h
~~~

~~~math
F_{\mathrm{attn}}\approx 4B H_qT_qT_kd_h
~~~

因此，self-attention 的主要 pair 计算项是 $O(BH_qT^2d_h)$。这个式子能说明序列长度加倍时该项约增至四倍；它不等于整层计算量，更不直接给出训练 step 时间。

训练一个 Transformer block 时，attention projection 和 FFN 也会做大量随 $T$ 线性增长的矩阵乘。以标准 FFN $d_{\mathrm{model}}\to d_{\mathrm{ff}}\to d_{\mathrm{model}}$ 为例，其两次矩阵乘约为 $4Td_{\mathrm{model}}d_{\mathrm{ff}}$ FLOPs；标准多头 attention 的 $QK^\top$ 与 $PV$ 约为 $4T^2d_{\mathrm{model}}$ FLOPs。只比较这两项时，交点约在 $T=d_{\mathrm{ff}}$，但 Q/K/V/O projection、其他 block 运算、硬件利用率和序列形状都会改变整层瓶颈。不能笼统说 attention 在任何长度、任何模型中都最慢。

显存需要另加一个条件：若实现显式物化 score 或 probability 矩阵，单个张量的字节数约为：

~~~math
M_S=B H_qT_qT_k\,s
~~~

$s$ 是该张量每个元素占用的字节数。例：$B=1$、$H_q=32$、$T_q=T_k=32768$，若每个元素按 BF16 的 2 bytes 估算，一张 $S$ 或 $P$ 约占 $64$ GiB；若两张同时物化，约占 $128$ GiB。这里仅计算矩阵 payload，不包括其他中间状态、对齐和框架开销。

这个例子说明朴素 materialized 路径为何困难，却不能推出 attention 本身必然需要平方级额外内存。流式累积、重计算和 FlashAttention 都能避免保存完整的 $T\times T$ 矩阵；在固定 head dimension 时，attention 的输入、输出和行统计随 $T$ 线性增长，而 pair 计算仍然是二次级。[Rabe 与 Staats](https://arxiv.org/abs/2112.05682)给出低于二次额外内存的 attention 算法，[Dao 等人的 FlashAttention](https://arxiv.org/html/2205.14135v2)则分析了分块 exact attention 的 IO 与 workspace。读到“attention memory 是 $O(T^2)$”时，应确认它说的是 pair 数、显式中间张量，还是整次训练的总显存。

## 10.3 FlashAttention 怎样分块又保持同一结果

只把矩阵乘拆成小块还不够。softmax 的分母需要覆盖同一个 query 可见的所有 key；如果每块单独 softmax，再把局部结果平均，归一化权重就会错。

FlashAttention 的 dense 路径把 Q/K/V 分块送入片上 SRAM，逐块更新每行的 softmax 统计和输出累积，避免把完整 score/probability 矩阵往返写入 HBM。在线 softmax 的状态是每个 query 行的当前最大值 $m$、归一化质量 $\ell$ 和加权和值向量 $u$。当前块中 $s_j$ 表示已经加上 mask 的缩放点积分数。初始化 $m=-\infty$、$\ell=0$、$u=0$；每次读入含有效 key 的新块后，按以下规则更新：

~~~math
m'=\max\left(m,\max_j s_j\right),\qquad
a=\exp(m-m'),\qquad
p_j=\exp(s_j-m')
~~~

~~~math
\ell'=a\ell+\sum_j p_j,\qquad
u'=au+\sum_jp_jv_j
~~~

更新后令 $(m,\ell,u)\leftarrow(m',\ell',u')$。所有 key 块处理完后，输出为 $o=u/\ell$。向量化实现会为 query block 的多行并行维护这些统计；这里按一行写，便于看清状态含义。若某个 tile 对该行完全不可见，就跳过该 tile；假设每个有效 query 至少能读取一个 key。

为什么以前累积的内容要乘 $a$？新的 tile 可能带来更大的 score 最大值 $m'$。为了让指数仍处在数值稳定范围，旧分数和旧加权和必须统一减去新的最大值；因而旧总量同时乘上 $\exp(m-m')$。课程材料 [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)从这个归一化问题推出分块算法；UC Berkeley 的 [Fall 2026 Attention and MoE Kernels 讲义](https://scalable-ai.eecs.berkeley.edu/F2026/assets/lecture_slides/lecture6.pdf)则用两块加权值说明为什么不能对局部平均值直接取平均。数学依据仍以 FlashAttention 原论文为准。

以下教学数值只用于追踪合并，不代表模型实测：块 A 的未归一化权重为 $1,1$，对应 value 为 $0,2$，其权重和为 $2$、局部加权平均为 $1$；块 B 的权重为 $6$，value 为 $4$，权重和为 $6$、局部加权平均为 $4$。全局结果应按权重和合并：

~~~math
o=\frac{2\cdot1+6\cdot4}{2+6}=3.25
~~~

把两个块的局部平均值直接平均会得到 $2.5$。$(m,\ell,u)$ 状态正是以数值稳定的方式保留合并所需的“质量”和“加权和”。

训练反向传播时，FlashAttention 可保存每行的归一化统计，并在 backward 重算当前 tile 的 score/probability，再累积梯度。这样以额外计算换取较少的 attention 中间存储。它不意味着整个 Transformer 训练只占线性显存：权重、梯度、optimizer state、跨层激活及其他 buffer 仍要计入。一般 activation checkpointing 也用重计算换内存，但作用范围不只 attention。FlashAttention 的 dense 核心计算同一 full softmax attention 数学算子，浮点累加顺序不同仍可能产生舍入差异；它不承诺逐位相同。原论文单列的 block-sparse 扩展改变了连接图，不属于这里的 dense exact 路径。

因此，FlashAttention 将“需要在 HBM 中反复物化的二次中间矩阵”改为分块计算和较小的行统计，并降低相应的数据搬运；它没有删除 full attention 的 $T^2$ 个 query-key 交互。GPU 版本、tile 调度和 API 支持见[第二十章](20-FlashAttention与PagedAttention.md)。

## 10.4 Causal mask 改变可见对数，但不自动减半资源

长度为 $T$ 的 causal self-attention 允许 query 位置 $i$ 读取位置 $0$ 到 $i$，理论可见 pair 数为：

~~~math
N_{\mathrm{causal}}=\frac{T(T+1)}{2}
~~~

它约为双向 full attention 的一半，阶数仍为 $O(T^2)$。但“有效可见 pair 减半”不能直接改写为“显存或运行时间减半”：若实现仍生成完整 dense score 张量，形状仍是 $T\times T$；支持 causal tile 跳过的 kernel 才可能少算被遮蔽区域。实际收益还取决于 mask pattern、shape、dtype、kernel 和硬件。若没有目标实现的证据，只报告可见 pair 数，不推断已节省多少时间或显存。

## 10.5 训练、prefill 与 decode 的长度关系

三个阶段需要的 query/key 长度不同，因此不能把一个 self-attention 公式直接当作整条生成链路的成本。

| 阶段 | 典型长度 | attention 位置交互 | 需要区分的瓶颈 |
| --- | --- | --- | --- |
| 训练 | $T_q=T_k=T$ | 一次并行计算整段序列；causal 模式只允许下三角位置可见 | 每步计算、保存的 activation 与训练总显存 |
| Prefill | $T_q=T_k=n$ | 一次处理 prompt，建立后续 decode 所需的 KV | prompt 变长会放大 pair 工作量；TTFT 还受排队、批处理和调度影响 |
| Decode 单步 | $T_q=1,\ T_k=t$ | 当前 query 读取长度为 $t$ 的历史 | 每步 pair 数随历史长度增长，且读取历史 KV；KV 容量与带宽见[第十一章](11-kv-cache长上下文推理和显存增长问题.md) |

如果 prompt 长度为 $n$，并将 $m$ 次单步 query 的 key 长度近似记为 $n,n+1,\ldots,n+m-1$，这些 decode query 的 pair 数合计为：

~~~math
N_{\mathrm{decode}}\approx B H_q\left(mn+\frac{m(m-1)}{2}\right)
~~~

该式数的是 attention 位置交互，不含历史 K/V 的实际字节布局、batch 中请求的长度差异和 runtime 调度。训练与 prefill 的 pair 结构相似，但训练另有 backward；decode 的 query 行较少，却反复读取历史状态。

## 10.6 不同优化分别改变了什么

“高效 attention”不是单一算法类别。判断一个方案时，先问它改了连接图、近似了算子，还是只改变计算与数据存储方式。

| 路线 | 改变的对象 | 典型长度成本（其他维度固定） | 读者要核对的边界 |
| --- | --- | --- | --- |
| Dense full attention | 保留所有允许的 query-key pair | $O(T^2d_h)$ | 计算量与中间矩阵是否物化是两个问题 |
| Local/sparse attention | 只连窗口、块或选定的全局位置 | 固定窗口 $w$ 时约为 $O(Twd_h)$，额外全局连接会增加相应 pair | 可见模式已改变；真实速度还取决于稀疏布局和 kernel |
| Low-rank sequence projection | 以低秩表示压缩序列交互 | 固定 rank 时可随 $T$ 线性增长 | 近似假设、rank、任务和误差范围 |
| Kernel feature map | 以有限维特征近似 softmax kernel，并重排聚合 | 固定特征维度时可随 $T$ 线性增长 | 近似性质、数值方差、因果实现和任务结果 |

Longformer 用 local window 加任务相关的 global token 构造稀疏连接图；[Linformer](https://arxiv.org/abs/2006.04768)以低秩近似缩短序列交互；[Performer](https://arxiv.org/abs/2009.14794)用 FAVOR+ 随机特征近似 softmax kernel。这些工作支持各自的方法定义与论文设置，不证明某一条路线对所有任务更好。更多局部/稀疏模式见[第十九章](19-Sliding-Window与稀疏注意力.md)；线性、核近似和替代序列架构见[第二十四至二十八章](24-Linear-Attention基本思想与核技巧.md)。

复杂度式只能帮助提出问题。比较真实实现时，还要固定模型、任务、mask、序列分布、dtype、硬件、kernel/backend 与 benchmark 口径；比较质量时另报长距离读取、复制、聚合或生成任务。较小的 Big-O 不等于在目标硬件上更快，也不保证近似路径保留了当前任务需要的信息。

## 10.7 复盘本章的成本账本

遇到 attention 性能或显存问题，可以按以下顺序拆分：

1. 从 $[B,H_q,T_q,T_k]$ 计算 query-key pair 数，确认是 self-attention、cross-attention 还是单步 decode。
2. 计算 attention 两个主要矩阵乘的 FLOPs；明确没有把 projection、softmax 和其他 block 计算混入。
3. 只有在目标实现物化了 $S/P$ 时，才按其 dtype 和 shape 估算对应中间矩阵显存。
4. 将 causal 可见边数量与 kernel 实际执行的 tile 分开；没有 profile 就不把可见边数当成时间节省。
5. 区分优化层次：FlashAttention 优化 dense attention 的分块、IO 与 workspace；稀疏/局部改变连接；低秩/特征映射改变算子或近似；PagedAttention 的分页 KV 管理由第二十章主讲，GQA/MQA 的 KV 结构由第四和第十一章主讲。

序列长度从 $T$ 增到 $8T$，dense pair 项增至 $64$ 倍；这足以说明长上下文为什么昂贵，却不能单独预测可运行的 batch、训练耗时、TTFT 或模型有效上下文能力。进入下一章后，讨论对象从 prompt attention 的 pair 计算转到自回归生成期间逐步积累的 KV cache 与显存容量。
