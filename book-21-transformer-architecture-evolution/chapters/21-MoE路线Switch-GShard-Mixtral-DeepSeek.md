# 第 21 章 MoE 路线：Switch、GShard、Mixtral 与 DeepSeek-MoE

## 21.1 先看 dense FFN 的瓶颈

在标准 decoder-only Transformer 中，attention 负责在 token 之间交换信息，FFN 或 MLP 负责对每个 token 做逐位置的非线性变换。一个 dense FFN 对所有 token 使用同一组参数：

~~~math
\mathrm{FFN}(x)=W_2\,\sigma(W_1x).
~~~

这种设计简单、规整，适合矩阵乘法和数据并行。问题是，模型想增加知识容量时，通常只能把 `W_1` 和 `W_2` 一起做大。参数量、每 token 的计算量、权重显存和训练成本于是同时上升。

语言、代码、数学和不同领域文本并不一定需要完全相同的变换。一个 token 可能更需要代码模式，另一个 token 更需要自然语言句法。如果所有 token 都经过同一个大 FFN，模型无法在计算预算不变的情况下增加很多不同的参数子空间。

Mixture of Experts（MoE）提出的基本想法是：保留一个共享主干，再准备多个 FFN expert；每个 token 只由 router 选择少数 expert 处理。这样模型可以拥有较大的总参数容量，而单个 token 的激活计算不必与 expert 总数同比增长。

这里的“专家”不是人工指定的学科模块，也不意味着某个 expert 永远只懂数学。expert specialization 是训练后可能出现的统计分工，必须通过路由分布、消融和任务切片验证，不能从 expert 编号直接推断语义。

## 21.2 从稀疏门控到现代 MoE

MoE 的历史不是一次完成的替换，而是一个问题逐步暴露、逐步修补的过程。

早期稀疏门控模型证明了条件计算可以把更多参数放进模型，但 router、容量限制和跨设备 dispatch 很快成为主要工程难点。GShard 把稀疏专家与自动分片结合起来，说明 MoE 的关键不只是网络公式，还包括如何把 token 发到拥有对应 expert 的设备。

Switch Transformer 进一步把路由简化为 top-1：每个 token 只选择一个 expert。top-1 减少了 combine 和通信压力，也减少了每个 token 被复制到多个 expert 的内存开销，但它更依赖负载均衡和容量管理。Switch 的经验表明，低精度训练、router 稳定化和容量设计必须一起处理。

Mixtral 采用稀疏 top-2 路由，把每个 token 送到两个 expert，再按 router 权重合并结果。它在公开权重生态中展示了“总参数容量”和“激活计算”可以分开讨论，也让工程师能直接观察 expert parallel、token dispatch 和推理吞吐的代价。

DeepSeekMoE 走向更细粒度的 expert 切分，并引入共享 expert 的思路。细粒度切分让 router 可以用更多小 expert 组合出更细的容量分配，共享 expert 则承担跨领域的共同知识，减少每个 routed expert 都重复学习通用模式的压力。DeepSeek-V2 又把 DeepSeekMoE 与 MLA 结合，说明现代模型优化常常同时处理 FFN 容量和 KV cache，而不是只改一个模块。

这些路线的共同问题可以概括为四个词：选择谁、装得下吗、发得过去吗、真的更好吗。后文会分别回答这四个问题。

## 21.3 Router、expert 和 combine 的完整数据流

设一个 batch 中有 `T` 个 token，hidden size 为 `d`，expert 数为 `N`。router 对每个 token 产生 logits：

~~~math
r_t=W_rh_t,\qquad
p_t=\mathrm{softmax}(r_t),
~~~

其中 `p_t` 是 token `t` 对 `N` 个 expert 的概率分布。top-k 路由选出集合 `S_t`：

~~~math
S_t=\mathrm{TopK}(p_t,k).
~~~

每个被选中的 expert 对 token 做自己的 FFN：

~~~math
y_t=\sum_{e\in S_t}\tilde p_{t,e}E_e(h_t),
\qquad
\tilde p_{t,e}=\frac{p_{t,e}}{\sum_{j\in S_t}p_{t,j}}.
~~~

`E_e` 是第 `e` 个 expert，`\tilde p` 是选中 expert 之后重新归一化的权重。实现时，系统不会逐 token 调用 Python 函数，而是把 token 按 expert 分桶，执行批量矩阵乘，再按原 token 顺序 combine。

整个路径可以写成：

~~~text
hidden states
  -> router logits
  -> top-k expert ids and weights
  -> token dispatch / all-to-all
  -> batched expert FFN
  -> combine by router weights
  -> restore original token order
~~~

因此，MoE 的理论计算量不能只看 expert FFN。真实运行时间还包括 router、排序或 permutation、跨设备通信、padding、combine 和同步。

## 21.4 容量因子为什么决定 token 是否被丢弃

如果每个 expert 都无限接收 token，热点 expert 可能把显存和计算队列撑爆。工程实现通常给每个 expert 一个容量：

~~~math
C=\left\lceil
\mathrm{capacity\_factor}\cdot\frac{T\,k}{N}
\right\rceil.
~~~

当一个 expert 收到的 token 数超过 `C` 时，系统需要选择 drop、reroute、padding 或扩大容量。不同实现的处理方式不同，不能把“capacity overflow”简单理解成模型输出一定错误；但 overflow 增多通常意味着有效训练信号损失、质量波动或额外通信。

举一个手算例子。假设 `T=8`、`N=4`、`k=2`，总 dispatch 次数是 16。若 `capacity_factor=1.0`，每个 expert 的容量为：

~~~math
C=\left\lceil\frac{8\times2}{4}\right\rceil=4.
~~~

如果 router 把 7 个 dispatch 都送到 expert 0，expert 0 只能接收 4 个，至少 3 个 dispatch 要被丢弃、改路由或等待下一种处理。平均负载看起来是 4，但平均值掩盖了热点。

提高 capacity factor 可以减少 overflow，却要付出 padding、显存和通信代价。降低 capacity factor 可以提高资源利用率，却可能使稀有但重要的 token 更容易被截断。它不是一个越大越安全的参数，而是质量、容量和尾延迟之间的验收条件。

## 21.5 负载均衡：不要只看 router loss

MoE 通常会加入辅助负载均衡目标，使不同 expert 接收的 token 数和 router 概率不要过度集中。教学化地写，可以令 `f_e` 表示实际分配给 expert `e` 的 token 比例，`P_e` 表示 router 对 expert `e` 的平均概率，则一种常见形式是：

~~~math
\mathcal{L}_{\mathrm{aux}}
=N\sum_{e=1}^{N}f_eP_e.
~~~

这个公式表达的是“实际流量”和“软概率”之间的匹配关系，具体系数、归一化方式和实现要以对应论文或代码为准。还可以监控：

1. `tokens_per_expert`：每个 expert 收到多少 token；
2. `probability_mass_per_expert`：router 概率质量如何分布；
3. `overflow_rate`：容量不足的比例；
4. `expert_utilization`：expert 计算资源实际利用率；
5. `route_entropy`：router 选择是否过于尖锐或过于平均；
6. `quality_by_route`：不同路由桶的任务质量。

辅助 loss 下降不等于模型和服务都健康。router 可能在 token 数上很均衡，却把高价值任务分给不合适的 expert；也可能为了追求均匀而破坏有意义的 specialization。因此质量、负载、通信和容量必须一起看。

## 21.6 一个最小 router 审计例子

下面的纯 Python 例子不训练模型，只展示 top-k 分配和容量审计。它故意把一个热点 router 的结果写出来，帮助读者看到“top-k 正常返回”和“容量可用”是两件事。

~~~python
from collections import Counter


def route(expert_scores, top_k, capacity_factor):
    token_count = len(expert_scores)
    expert_count = len(expert_scores[0])
    capacity = (capacity_factor * token_count * top_k + expert_count - 1) // expert_count
    assignments = []
    for token_id, scores in enumerate(expert_scores):
        chosen = sorted(range(expert_count), key=lambda e: scores[e], reverse=True)[:top_k]
        assignments.extend((token_id, expert_id) for expert_id in chosen)

    counts = Counter(expert_id for _, expert_id in assignments)
    overflow = {e: max(0, counts[e] - capacity) for e in range(expert_count)}
    return capacity, counts, overflow


scores = [
    [0.9, 0.1, 0.0, 0.0],
    [0.8, 0.2, 0.0, 0.0],
    [0.7, 0.3, 0.0, 0.0],
    [0.6, 0.4, 0.0, 0.0],
]
print(route(scores, top_k=1, capacity_factor=1.0))
~~~

这个例子中所有 token 都选择 expert 0。真实实现还要处理概率权重、drop policy、跨设备通信和 batch 内 padding，但这个最小审计已经能发现 router collapse 的最直观形态。把它扩展到训练日志时，应该按 step、层、语言、任务类型和序列长度保存 histogram，而不是只保留一个 epoch 平均值。

## 21.7 为什么 MoE 往往替换 FFN 而不是 Attention

FFN 对每个 token 独立计算，天然适合把不同 token 分发到不同 expert。attention 则要在 token 之间建立交互，路由 attention 本身会改变信息可见性、KV cache 和并行通信，工程复杂度更高。

因此许多 Transformer MoE 模型把 dense FFN 换成 MoE FFN，同时保留 attention 主干。这样可以让 token 在 attention 中共享上下文，再在 FFN 中进行条件化变换。

这不是理论上的硬限制。研究者可以设计 expert attention、混合层或 request-level router，但每增加一层动态路由，就要重新审计 mask、cache、通信和训练稳定性。面试中更稳妥的表达是：MoE 最常见的落点是 FFN，因为 FFN 是逐 token 的容量模块，路由边界清楚、并行实现相对成熟。

## 21.8 Expert parallel 与 all-to-all

当专家分布在不同 GPU 上时，token 需要先从当前设备发送到目标 expert 所在设备，expert 计算完成后再把结果发回。这个过程通常包含 dispatch all-to-all 和 combine all-to-all。

设每层 dispatch 的 token 数为 `T_k=T\times k`，每个 token 的 hidden 向量使用 `b` 字节，设备间通信量的教学近似为：

~~~math
V_{\mathrm{comm}}\propto T_k\,d\,b.
~~~

实际通信量还取决于 expert placement、是否有本地 expert、padding、拓扑和 collective 实现。增加 expert 数可以增加总容量，却可能把更多 token 发到远端；top-2 可能提高组合表达力，却也可能近似翻倍 dispatch 负担。

GPU 利用率低不一定是矩阵乘不够快，也可能是 token 在等待 all-to-all；某个 expert 的计算时间长也不一定是它的参数更多，可能是路由热点造成 batch 不均。MoE profiling 要把 router、通信、expert kernel 和 combine 分开计时。

## 21.9 训练和推理的不同问题

训练时，一个 batch 中 token 数通常较多，expert 可以形成较大的矩阵乘，通信也更容易被计算隐藏。但训练要面对 router 梯度、负载均衡、容量 overflow、低精度数值稳定和专家 collapse。

解码时，每个请求每一步生成的 token 很少，动态 batch 可以把多个请求合在一起，但路由分布会随内容和时间变化。短请求、长请求和不同采样路径可能在同一个 step 产生高度不均匀的 expert load。此时 router overhead、dispatch latency 和 p99 可能比单个 expert 的 FLOPs 更重要。

训练可用不代表服务可用。训练中平均 token 数很大，推理中 batch 可能很小；训练可以容忍较高同步时间，交互式服务却可能无法接受一个热点 expert 拉长尾延迟。

## 21.10 细粒度 expert 与共享 expert 的理解方式

把 expert 切得更细，可以让 router 用多个小模块组合表示。直觉上，大 expert 像几个大的工具箱，细粒度 expert 像更多小工具，选择空间更灵活；代价是 router 的选择问题更难、dispatch 元数据更多、专家之间的参数更新可能更稀疏。

共享 expert 可以看作每个 token 都会经过的共同知识路径。它的作用不是保证某个 expert 学会“通用知识”，而是减少 routed experts 重复学习常见模式的压力。共享路径会增加每 token 的固定计算，因此必须和 active parameters、质量和通信一起核算。

阅读 DeepSeekMoE 时要把论文报告和普遍结论分开。论文可以支持该架构在其训练规模和实验设置中的结果；它不能自动证明所有任务、所有硬件和所有 MoE 都会因细粒度 expert 获益。

## 21.11 MoE 的失败模式

### Router collapse

router 长期把 token 集中到少数 expert，导致热点、overflow 和冷门 expert 学不到东西。排查要按层、step 和数据桶看 route histogram，不能只看最终 loss。

### Expert collapse

多个 expert 的表示和行为逐渐相似，模型付出了多份参数，却没有获得有效的条件容量。可以做 expert 输出相似度、替换 expert、冻结/重置局部 expert 和任务切片消融。

### Capacity overflow

容量过小会丢 token 或改变路径，容量过大则引入 padding 和显存浪费。要记录 overflow 的 token 类型和位置；如果高价值任务更容易 overflow，平均 overflow rate 会掩盖实际质量损害。

### Communication hotspot

expert placement 与路由分布不匹配时，跨节点 all-to-all 会成为瓶颈。要把 expert load、跨节点字节数、collective 时间和网络拥塞关联起来。

### 训练与服务不一致

训练使用一种容量和 drop policy，服务使用另一种；训练按大 batch 路由，服务按小 batch 路由；训练忽略取消和 preemption，服务却必须支持。这些差异会让离线分数无法预测线上表现。

## 21.12 如何评估一个 MoE 方法

一个公平的 MoE 实验至少包含 dense baseline、相近 active FLOPs 的 MoE、相近总参数的 dense baseline，以及不同 expert 数和 top-k 的消融。固定 tokenizer、训练 token、优化器、学习率计划、数据 mixture、上下文长度和评估脚本。

质量指标包括通用能力、代码、数学、多语言、长上下文和安全；系统指标包括 active FLOPs、总权重显存、expert load 方差、overflow、all-to-all 时间、吞吐、TTFT、TPOT、p95/p99 和单位成功任务成本。

可以把一个简单的 MoE 验收条件写成：

~~~math
G_{\mathrm{MoE}}
=G_{\mathrm{quality}}
\land G_{\mathrm{load}}
\land G_{\mathrm{capacity}}
\land G_{\mathrm{communication}}
\land G_{\mathrm{serving}}.
~~~

其中每个子门都应有可复现阈值和 workload。不能因为一个公开 benchmark 提升了，就忽略专家热点或工具调用任务的副作用。

## 21.13 面试中如何回答“MoE 为什么有效”

先说机制：MoE 用 router 对 token 做条件计算，每个 token 只激活少数 expert，因此在相近每 token 计算下增加总参数容量。

再说代价：它引入路由、容量、token dispatch、all-to-all、负载均衡、专家并行和 serving 尾延迟问题。

最后说证据：要在相同 token、active compute、训练预算和硬件条件下比较质量、overflow、通信和端到端成本；不能只比较 total parameters 或单次 tokens/s。

常见追问包括：top-1 与 top-2 的差异、为什么需要 capacity factor、为什么 router loss 不够、expert parallel 如何通信、MoE 是否减少权重显存、为什么训练快不代表推理快。

## 21.14 资料范围与资源账本

本章的历史和基本机制参考：GShard（https://arxiv.org/abs/2006.16668）、Switch Transformers（https://arxiv.org/abs/2101.03961）、Mixtral of Experts（https://arxiv.org/abs/2401.04088）、DeepSeekMoE（https://arxiv.org/abs/2401.06066）和 DeepSeek-V2（https://arxiv.org/abs/2405.04434）。公式中的容量和通信关系是教学近似，具体 drop policy、router loss、expert placement 和 kernel 行为必须以对应论文、源码和硬件实测为准。

## 21.15 MoE 训练和 serving 的资源账本

训练 batch 通常有足够多的 token，expert 可以形成较大 GEMM；在线 decode 每个请求每步只有少量 token，路由和 all-to-all 的固定成本更突出。可以把单步服务时间拆成：

~~~math
T_{\mathrm{step}}
=T_{\mathrm{router}}+T_{\mathrm{dispatch}}
 +T_{\mathrm{expert}}+T_{\mathrm{combine}}
 +T_{\mathrm{sync}}.
~~~

增大 expert 总数主要提高容量，降低 `top-k` 主要改变 active compute 和通信，增大 capacity factor 主要改变 overflow 和 padding。它们对质量、显存、吞吐和 p99 的影响方向不完全一致。训练日志中的平均 FLOPs 不能替代 serving trace；小 batch 的热点 expert 可能让 `T_sync` 主导总延迟。

## 21.16 Expert load 的 p99 比平均值更接近事故

设一个 step 中每个 expert 收到的 dispatch 数为 `n_e`，平均值为 `bar n`。平均负载偏差可以写成：

~~~math
\mathrm{CV}_{\mathrm{load}}
=\frac{\sqrt{\frac{1}{N}\sum_e(n_e-\bar n)^2}}{\bar n+\varepsilon}.
~~~

但服务尾延迟往往由 `max_e n_e` 和跨节点通信路径决定，而不是 CV 单独决定。应同时记录最大/ p99 expert load、overflow token 类型、跨节点 bytes、collective duration、冷 expert 命中和 fallback。若热点集中在代码或某种语言，整体平均值会掩盖业务切片的质量损失。

一个有效的诊断顺序是先固定 router，重放同一 dispatch 流，测 expert kernel 和通信；再固定 expert，替换路由分布，测质量和 overflow；最后才改变模型结构。这样可以区分“expert 本身慢”“路由不均”“拓扑不匹配”和“容量策略不合适”。

## 21.17 router 的训练目标和业务质量可能冲突

负载均衡鼓励 token 分散，specialization 则可能希望某些任务集中到少数 expert。均衡 loss 太强，可能迫使相似 token 走不同路径；过弱则产生热点和冷 expert。因而 router 目标不应只看 auxiliary loss，而要和任务质量、overflow、路由熵及 expert 输出差异一起分析。

可以做三种反事实：固定 expert 输出，只打乱 router；固定 router，只交换两个 expert 的参数；把 router 概率温度改变后重放同一 batch。若打乱 router 对高价值任务影响很大，说明路由承担了实际分工；若只改变 auxiliary loss 就改变 p99 而质量不变，可能应优先优化 runtime；若 expert 交换几乎无影响，说明 specialization 证据不足。

## 21.18 MoE 的 checkpoint、迁移和回滚

MoE checkpoint 除了权重，还包含 expert placement、router 配置、capacity/drop policy、通信拓扑假设和可能的 optimizer state。迁移到不同 GPU 数量或节点拓扑时，不能只重新加载权重就认为行为完全一致；expert 重排、量化和路由 tie-break 都可能改变 batch 路径。

服务侧的 speculative decoding、取消和 preemption 还要区分已接受 token 的 expert state、临时 dispatch buffer 和 page/block 资源。被拒绝的候选不能留下 expert 输出或引用计数；请求取消后，跨设备通信完成顺序也不能让晚到的结果写入复用的 batch slot。回滚记录应包含 request id、token offset、expert assignments、capacity decision 和 model revision。

## 21.19 DeepSeek 路线应按公开证据分层

DeepSeekMoE、DeepSeek-V2/V3 等公开论文可以支持其中披露的细粒度 expert、共享 expert、MLA、训练和评测设定；模型产品的完整 kernel、数据处理、调度策略和所有后训练细节不应由名称推断。Mixtral、Switch、GShard 与 DeepSeek 的对比也要区分论文实验和第三方部署观察。

更稳妥的阅读方式是建立字段表：total/active 参数、expert 数、top-k、共享路径、capacity/drop、训练 token、通信拓扑、KV/latent cache、评测 harness 和未知字段。只有字段和证据等级都对齐，横向结论才不会把“某模型使用 MoE”扩写成“所有 MoE 都获得同样收益”。

## 21.20 一个 MoE 发布条件

可以把 MoE 进入生产的验收条件写为：

~~~math
G_{\mathrm{MoE}}
=G_{\mathrm{quality}}\land G_{\mathrm{overflow}}
 \land G_{\mathrm{communication}}\land G_{\mathrm{p99}}
 \land G_{\mathrm{isolation}}\land G_{\mathrm{rollback}}.
~~~

其中质量门覆盖代码、数学、多语言、长上下文、工具和安全；overflow 门覆盖 token 丢失与 fallback；communication 门覆盖跨节点字节和 collective p99；isolation 门覆盖跨请求/租户状态；rollback 门覆盖候选路由和已提交 token。单个 benchmark 提升不能绕过这些硬性条件。

## 21.21 小结

MoE 的核心贡献不是把模型简单拆成很多 FFN，而是把“模型容量、每 token 计算和设备资源”分成可以分别优化的账本。Router 决定信息走哪条路径，capacity 决定路径能否接收，expert parallel 决定 token 能否及时到达，评测决定这种复杂性是否换来了真实任务收益。
