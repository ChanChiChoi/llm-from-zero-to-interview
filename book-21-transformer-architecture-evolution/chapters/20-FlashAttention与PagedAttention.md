# 第二十章：FlashAttention、PagedAttention 与 Attention Kernel 优化

## 20.1 先分清四类优化对象

第十章统计 attention 要计算多少 query-key 位置对，并解释 online softmax 怎样跨块归一化；第十一章估算生成期间每个请求保存多少 KV。这里继续追问：这些计算和状态在 GPU 上怎样移动、分工和寻址？

| 方法 | 改变什么 | 是否改变原 attention 的可见位置 | 主要缓解的问题 |
| --- | --- | --- | --- |
| FlashAttention | 把完整 attention 按 tile 计算，减少 score/probability 中间量在 HBM 与片上存储之间的往返 | 否 | 物化中间矩阵的数据搬运和额外存储 |
| Flash-Decoding | 将 decode 时较长的 K/V 维度分段并行计算，再合并各段结果 | 否 | 单步 query 行少，GPU 并行度不足 |
| PagedAttention | 用 block table 将逻辑 KV block 映射到不连续物理 block | 否 | 动态 KV 分配的连续预留浪费及前缀状态共享 |
| Block-sparse attention | 跳过被 mask 排除的 query-key block | 是 | 计算较少的位置对；布局和 kernel 另有约束 |

前三种改变计算的组织方式或 K/V 的存放与寻址；block-sparse 则改变 attention 连接图，见[第十九章](19-Sliding-Window与稀疏注意力.md)和[第二十六章](26-稀疏与块稀疏Attention.md)。把它们统称为“attention 加速”会遮住各自的成本和语义边界。

## 20.2 FlashAttention：完整 attention 怎样减少中间数据往返

GPU 把 Q/K/V 和中间结果放在高带宽显存（HBM）中；寄存器与 shared memory 等片上存储离计算单元更近，但容量有限。kernel 是 GPU 上执行的一段计算程序。直接 attention 通常先算所有允许的 score，再对每行做 softmax，最后用概率加权汇总 value。若把 score 和 probability 都写入 HBM，两个中间矩阵在多个计算阶段之间来回读写；它们的 shape 和 pair 数已在[第十章](10-o-n2-attention的计算与显存瓶颈.md)说明。FlashAttention 的观察是，这些中间结果不必先存成完整的 $N\times N$ 矩阵。

FlashAttention 按 tile 处理 Q、K、V。针对同一批 query 行，kernel 逐块读取对应的 K/V，计算局部 score，并把局部 softmax 结果并入每行的累计状态：

| 每行状态 | 含义 | 后续用途 |
| --- | --- | --- |
| $m$ | 已处理 score 的最大值 | 为指数运算保持数值稳定 |
| $\ell$ | 相对该最大值的指数权重和 | 保留全局 softmax 的归一化分母 |
| $u$ | 同一套指数权重对 V 的加权和 | 保留 attention 输出的分子 |

处理新 tile 时，局部最大值可能改变 $m$，因此旧的 $\ell$ 与 $u$ 必须按新的最大值重标定，再合入当前 tile。所有有效 K/V 块处理后，以 $u/\ell$ 得到该行输出。这个状态沿用第十章的在线 softmax 推导；本节关注的是：每块短暂留在片上完成计算，而不把完整 score/probability 写回 HBM。原始 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)给出 exact attention 算法和 IO 分析；Berkeley 的[课程讲义](https://scalable-ai.eecs.berkeley.edu/F2026/assets/lecture_slides/lecture6.pdf)用 S/P 中间矩阵的 HBM 往返图示说明了问题。

这里的 exact 表示它仍对同一个可见集合计算标准 softmax attention，没有用稀疏图或核近似替代 dense 连接。tile 顺序改变后浮点加法顺序也会变化，所以结果允许有舍入误差，不承诺逐位一致。训练 backward 可以重新计算 tile 的 score/probability，减少需要保存的中间结果，但代价是重复计算；权重、optimizer state 和其他层 activation 仍计入训练显存。

FlashAttention 因而减少的是 attention 中间数据的 HBM 流量和额外存储，不会把 dense 的 query-key pair 计算从二次量级变成线性量级。若原实现已经受其他矩阵乘、短序列启动开销或服务调度限制，IO 改善也不必然成为同等幅度的端到端加速。

## 20.3 Kernel 演进：并行划分、硬件流水与 decode

FlashAttention 的版本演进主要是对不同 GPU 上的瓶颈重新安排工作，而不是改变 softmax attention 的数学定义。

| 路线 | 针对的瓶颈 | 主要变化 |
| --- | --- | --- |
| FlashAttention（2022） | 中间矩阵反复读写 HBM | 使用片上 tiling 和在线 softmax，减少中间矩阵物化 |
| FlashAttention-2（2023） | thread block/warp 工作分配不均、共享内存通信和非矩阵乘操作 | 沿序列维增加并行工作，调整 warp 分工；论文的效率数据来自其 A100 等测试配置 |
| FlashAttention-3（2024） | Hopper Tensor Core 与数据搬运能力未充分重叠 | 使用异步调度与 warp specialization 交叠搬运和矩阵计算，并研究 FP8 路径；H100 结果及数值误差比较绑定论文设置 |
| FlashAttention-4（2026） | Blackwell 上矩阵计算吞吐与共享内存、指数运算等资源的扩展不对称 | 重新设计异步矩阵流水、tile 调度和 softmax 相关计算；性能结论绑定 B200、dtype、长度及对照库版本 |

这些层次的变化见 [FlashAttention-2 原论文](https://arxiv.org/abs/2307.08691)、作者在 [Stanford CRFM 的说明](https://crfm.stanford.edu/2023/07/17/flash2)、[FlashAttention-3 原论文](https://arxiv.org/abs/2407.08608)和 [FlashAttention-4 原论文](https://arxiv.org/abs/2603.05451)。以 FA4 为例，论文将测试固定在 B200、FP16/BF16、指定长度与 head shape，并和 cuDNN 9.13、Triton 等基线比较；同一论文还注明后续 cuDNN 版本吸收了相关技术后表现接近 FA4。因此该数字只能说明对应实验里的实现比较，不能当作当前库或所有 GPU 的固定排名。

训练与 prefill 往往有较多 query 行可并行；decode 每一步通常只有一个或少量新 query，却要读取不断增长的历史 K/V。若只沿 query 行并行，batch 较小时可用工作可能不足。Flash-Decoding 将 K/V 序列切成若干分段，在各段上并行计算 query 的局部 attention。每段返回与 20.2 节相同类型的行统计 $(m_j,\ell_j,u_j)$：$m_j$ 是该段 score 最大值，$\ell_j$ 是相对该最大值的指数和，$u_j$ 是对应的加权 value 和。归并时先统一各段的指数参考点，再相加分母与分子：

```math
m = \max_j m_j,\qquad
\ell = \sum_j \exp(m_j-m)\ell_j,\qquad
u = \sum_j \exp(m_j-m)u_j,\qquad
o = \frac{u}{\ell}.
```

因此分段结果不能直接做算术平均；按归一化统计加权后，才得到对全部允许 key 的同一次 softmax。这个拆分与归并过程见 [PyTorch 团队的 Flash-Decoding 说明](https://pytorch.org/blog/flash-decoding/)。它增加了一次跨分段归并，适用收益取决于上下文长度、并行规模和实际 kernel；它没有删掉被允许的历史位置，也不是 KV cache 分页本身。

## 20.4 PagedAttention：逻辑连续的 KV 与物理分散的 block

自回归生成的 K/V 随 token 逐步增加。如果系统为每条请求预留最大长度的一整段连续显存，短请求会占着尚未使用的位置；请求长度变化也让共享和回收困难。PagedAttention 把每条逻辑 KV 序列切成固定 token 数的 block，再用 block table 将逻辑位置映射到物理显存 block。相邻逻辑块可以放在物理上互不相邻的位置。

设请求 $r$ 的 token 位置从 $0$ 开始，block 容量为 $B$。位置 $t$ 落在第 $b$ 个逻辑块内的偏移 $o$，该逻辑块对应的物理块编号为 $p$：

```math
b=\left\lfloor\frac{t}{B}\right\rfloor,\qquad
o=t\bmod B,\qquad
p=P_r[b].
```

其中 $P_r$ 是请求 $r$ 的 block table。比如 $B=4$ 时，逻辑块 $0$ 可以指向物理块 $12$，逻辑块 $1$ 指向物理块 $3$；逻辑顺序仍是 $0$ 至 $7$。这些物理编号仅用于说明映射关系。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180v1)说明了 logical block、physical block 和 block table 的设计。

decode kernel 沿请求的 table 找到每个 K/V block，按逻辑 token 顺序读取；分段 score 仍要对同一 query 的全部有效 key 做全局 softmax 归一化。把所有物理块先复制、拼成新的连续大张量再 attention，会重新引入分页原本要避免的数据移动。分页降低最大长度预留浪费并允许按需增长，不会减少每个活跃 token 的逻辑 KV payload；每条请求未填满的尾块仍可能留下最多 $B-1$ 个 token 槽位。块越小，尾部空槽上限越低，但同样数量的 token 需要更长的 block table；块越大则相反，因此 block size 是元数据、尾块浪费和访问成本之间的实现取舍。

地址查找、变长处理和非连续访问也要付出成本。PagedAttention 原论文在其特定测试中报告，attention kernel latency 比优化过的 FasterTransformer 基线高约 20%–26%；论文也单独报告了更灵活的内存管理对系统并发与吞吐的作用。[原论文 §7.1](https://arxiv.org/abs/2309.06180v1)中的 kernel 差值不能当作当前 vLLM kernel 的常数，也不能据此否定或保证其他负载下的端到端收益。

## 20.5 共享前缀：复用 KV 并保护分支写入

并行采样、beam search 或重复 system prompt 会产生相同前缀。若前缀 KV blocks 完全相同且其计算条件兼容，多个请求可以在 block table 中引用同一组只读物理块，而不必各留一份副本。请求分叉后，新增 token 逐步写入自己的状态；若下一次写入会改动仍被其他请求引用的 block，就必须先复制该块或采用等价的隔离规则。引用计数归零后，物理块才可回到空闲池。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180v1)以 parallel sampling 说明了共享前缀、引用计数与 copy-on-write。

PagedAttention 提供了共享 block 的表示方式；自动 prefix cache 的 key、淘汰策略和适配器/多模态区分属于 engine 实现。例如 [vLLM 的 Automatic Prefix Caching 文档](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)按前缀和 block 内容识别可复用状态，并说明具体 KV cache group 与版本条件。复用前提是影响 K/V 的模型状态、token 序列、位置和相关配置一致；不兼容的 tokenization、位置规则或模型适配不能因为文本看起来相同就共享。活跃请求仍引用的 block 不能作为普通 cache 条目随意淘汰。

## 20.6 组合与后端选择：名称不等于执行路径

FlashAttention 与 PagedAttention 可以处理系统里的不同部分：前者计算 attention 时减少中间矩阵搬运，后者为动态 K/V 提供逻辑到物理的 block 映射。Flash-Decoding 还可以针对少量 query、长历史增加 K/V 维并行。是否能把分页读取、指定 FlashAttention 版本、mask 和某种 dtype 组合在一个 fused kernel 中，取决于具体后端是否支持那种布局和 shape；概念上互补不代表任一框架都会自动组合。

高层 SDPA API 也不能单独证明 kernel 路径。以 PyTorch 2.15 文档为例，SDPA 会在可用实现间选择，融合 kernel 有输入限制，不同后端的浮点结果可能有细微差异；需要某个特定 backend 时，可通过[受控选择入口](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.sdpa_kernel.html)限定路径，相关限制见 [SDPA API 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)。因此应按具体 PyTorch 版本、shape、dtype、mask 和设备解释实际路径，不凭 API 名称推断一定调用了 FlashAttention。

比较效果时，先区分 kernel latency、端到端吞吐与延迟、KV 占用和可接纳请求数。PagedAttention 的 block table 可能增加单次访问开销，却减少为最长长度预留的空间；更多请求能同时放入设备时，系统结果可能与单 kernel 的快慢不同。服务调度与端到端压测由相应推理系统资料主讲，参见[第六册第六章《批处理与调度》](../../book-06-llm-deployment/chapters/06-批处理与调度.md)和[第二十四册第五十三章《构建 serving engine 的 benchmark 指标和调参实验框架》](../../book-24-llm-inference-engine/chapters/53-构建serving-engine的benchmark指标和调参实验框架.md)。其他 KV 分配路线，如 [vAttention](https://arxiv.org/abs/2405.04437v3)，也采用不同的映射和内存管理机制；分页是一种设计选择，不是所有推理系统的前提。

因此，读到“用了 FlashAttention”时，追问它保留了哪些连接、实际采用哪个版本和 kernel；读到“用了 PagedAttention”时，追问 block table 和共享状态怎样维护、测量的是 kernel 还是系统结果。只有把算子计算、KV 寻址、设备版本和请求工作负载分别说清，才能判断一次优化究竟改变了什么。
