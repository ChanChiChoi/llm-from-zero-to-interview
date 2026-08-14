# 第二十六章：稀疏 Attention、局部 Attention 与块稀疏 Attention

## 26.1 稀疏不是一个单一算法

“稀疏 attention”可能指固定局部窗口、随机连接、global token、块稀疏、可学习路由或运行时 top-k。它们共同减少可计算的 query-key 对，却拥有不同的信息图、训练方式和硬件实现。

如果只把 dense attention 矩阵中的一部分元素设为零，GPU 仍可能按 dense 矩阵计算，理论 FLOPs 减少不会变成实际速度。真正的块稀疏需要 kernel 了解非零块布局，并让数据加载、线程和 warp 组织与 pattern 对齐。

## 26.2 token 级与 block 级稀疏

token 级 mask 可以定义任意 `M_{ij}`，表达细粒度连接；block 级 mask 将序列分成大小 `b` 的块，只在允许的 block pair 上计算：

~~~math
\mathcal{B}_{uv}=
\begin{cases}
1,&\text{block }u\text{ 可以读取 block }v\\
0,&\text{otherwise}
\end{cases}
~~~

若非零块比例为 `\rho`，理想 score 计算约为 `\rho T^2d`。实际还要加 block 内 padding 和索引成本。`b` 越大，kernel 越规则但细粒度控制越弱；`b` 越小，稀疏表达更细但管理开销更大。

## 26.3 局部、扩张和全局 pattern

局部 pattern 关注邻近 token，适合语法和连续信号；扩张窗口以更稀疏的间隔连接远处位置；global pattern 让少量 token 汇聚/广播全局信息。不同层可以使用不同窗口和 dilation，形成多尺度路径。

一个多尺度窗口可以写为：

~~~math
\mathcal{N}_l(i)=
\{i-jd_l\mid 0\le j<w_l\}
~~~

`d_l` 是第 `l` 层的 dilation。它在理论上扩大感受野，却可能造成 aliasing：某些关键邻近 token 从未被同一层直接看到。

## 26.4 块布局的最小计算

~~~python
import torch


def block_pairs(seq_len, block_size, allow):
    blocks = (seq_len + block_size - 1) // block_size
    pairs = []
    for q_block in range(blocks):
        for k_block in range(blocks):
            if allow(q_block, k_block):
                pairs.append((q_block, k_block))
    return pairs


def local_causal(q_block, k_block, window_blocks):
    return k_block <= q_block and q_block - k_block < window_blocks


print(block_pairs(10, 2, lambda q, k: local_causal(q, k, 2)))
~~~

这个 demo 只生成 block pair，不计算 Q/K/V。实际 kernel 还要处理最后一个不完整 block、causal 边界、batch 变长、global block 和梯度。

## 26.5 稀疏结构的图指标

把每层允许连接看作有向图，除了边数，还可以测：从早期证据到末端 query 的最短路径、图直径、可达节点比例、global token 的入度/出度和不同 head 的边重叠。

若一条证据在每层都能沿局部边传播，路径长度过长可能造成表示衰减；若依赖 global token，global 表示可能成为拥塞瓶颈。可用 reachability probe 检测，而不必只凭直觉讨论“全局信息是否存在”。

## 26.6 训练、mask 与因果性

稀疏 mask 要与训练数据和推理协议一致。非因果任务可以让一段输入双向交互；生成任务仍需禁止未来 token。prefix LM、encoder-decoder、图文交错和工具 observation 都可能需要不同 mask。

最危险的 bug 是训练时某个 global token 可以读全序列，推理时 position/role 变化；或实现把未来的 global block accidentally 打开，导致训练 loss 虚高或推理时泄漏。应在单个 batch 上打印允许矩阵，并用一个未来 token 反事实测试信息是否真的不可见。

## 26.7 稀疏与显存

理论上稀疏 score 减少显存，但 KV cache 是否减少取决于历史存储策略。局部 attention 如果只保存窗口，KV 可以固定；若为了后续 global 层仍保存全部历史，cache 收益有限。global token、prefix cache 和分页管理也会改变实际布局。

因此内存模型应写成：

~~~math
M_{\mathrm{KV}}=
M_{\mathrm{local}}+M_{\mathrm{global}}+M_{\mathrm{metadata}}
~~~

稀疏算法的 metadata、index 和 block table 不能忽略。只有在 `M_metadata` 相对较小、kernel 能按 pattern 读取时，理论节省才会落地。

## 26.8 可学习稀疏和输入依赖路由

固定 pattern 的优点是可预测、容易编译；可学习稀疏让模型根据内容选边，更灵活，但路由需要额外预测、排序和负载管理。token 级 top-k attention 可能产生不规则内存访问，训练梯度也更复杂。

如果路由选择了错误的远程证据，模型可能没有可恢复路径。需要为 route 做辅助损失或保留 dense/local fallback。稀疏率越高，质量回退边界越需要通过任务级消融确定。

## 26.9 与 MoE 的类比和区别

MoE 在 FFN 维度对专家做稀疏路由，稀疏 attention 在 token-token 关系上做稀疏路由。两者都有 routing、负载、通信和质量 trade-off，但 MoE 主要改变 value transformation，稀疏 attention 改变信息读取路径。不能用 MoE 的 active parameter 术语描述 attention edge。

## 26.10 失败模式与压测

常见失败包括 block 边界 off-by-one、最后块 padding、global token 泄漏、稀疏 kernel fallback、索引读取不合并、不同序列长度导致负载失衡、远距精确检索下降和训练/推理结果不一致。

压测要报告理论非零块比例、实际 FLOPs、kernel 时间、显存、tokens/s、TTFT、TPOT、p99、证据 recall 和多证据组合准确率。一个模型可能质量持平但尾延迟变差，也可能速度很好却丢掉少量高价值实体。

## 26.11 机制与边界：规则 pattern 与硬件编译

高效稀疏 kernel 往往要求固定 block size、静态 pattern 或有限模板。动态稀疏需要把路由结果排序、压缩、执行和还原，可能增加 kernel launch 和同步。编译器若无法融合 mask、GEMM 和 softmax，稀疏收益会被小矩阵操作吞掉。

研究评估应把算法层和实现层分开：先用统一 dense reference 测质量，再在同一硬件和相同 batch 上测 kernel，最后接入 serving scheduler 测真实请求。不要把论文中的渐近复杂度直接写成产品吞吐。

## 26.12 面试追问、误区与练习

**问：为什么 token 级稀疏不一定比 dense attention 快？**

标准回答：不规则索引、排序、同步和 kernel launch 可能抵消减少的 FLOPs；只有 pattern 规则、block 合理且硬件 kernel 支持时，理论稀疏才可能变成端到端 speedup。

**问：稀疏 attention 主要减少什么？**

标准回答：主要减少可计算的 query-key 交互；是否减少 KV cache、激活和带宽，取决于窗口、global 路径和实现，不能笼统回答“显存都减少”。

常见误区包括把 mask 置零当作真正稀疏、忽略路径长度、把局部窗口当作有效上下文，以及没有测试 future leakage。

练习：实现 local、dilated、global+local 三种 block mask，计算图直径和非零块比例，并在一个合成远距复制任务上比较。

## 26.13 稀疏 pattern 的训练一致性

固定稀疏 pattern 让 kernel 容易编译，但可能限制模型看到关键远程证据；输入依赖 pattern 提高灵活性，却使训练和推理的 mask、路由和 batch shape 更复杂。

训练时如果使用一种 pattern、推理时使用另一种 pattern，模型可能出现分布偏移。应做 mask parity test：对同一 token 序列比较训练 forward、离线推理和线上 kernel 的可见位置集合。

## 26.14 block size 的两面性

块越大，索引和 kernel 越简单，但块内会包含更多无关 token，实际稀疏率下降；块越小，理论访问更精确，却增加 metadata、launch 和调度开销。

一个简单的有效稀疏率为：

~~~math
\rho_{\mathrm{effective}}
=1-\frac{\mathrm{executed\ blocks}}{\mathrm{dense\ blocks}}
~~~

这个比例必须用真实执行 block 统计，不能只由 mask 中的零元素推算。

## 26.15 mask、block 和真实执行路径

块稀疏实现把逻辑 mask 映射为若干非零 block。若 block size 为 B，某个 block 内只需要一个 token 可见，整个 B×B 区域可能都被执行；逻辑稀疏率因此会高于硬件真正跳过的计算比例。

应同时记录逻辑非零边、执行 block 数、实际读写字节和 kernel fallback。只有 mask 中有很多零，不代表 GPU 没有读取这些零对应的值。

## 26.16 图直径和任务难度

把 sparse attention 看成层间有向图，可以估计远程信息需要经过多少层传播。若 global token 是唯一桥梁，它的表示容量、位置和训练信号会成为瓶颈；若随机边提供连通性，单次运行的 seed 可能改变具体证据路径。

评估要固定 mask seed，并比较不同 seed 的方差。对精确数字和多实体冲突任务，还要检查关键证据是否在每层都有可达路径，而不是只看理论图连通。

## 26.17 动态稀疏和 serving 调度

输入依赖的稀疏模式可能更适合当前 query，却会让 batch 中每个请求执行不同 block，增加 metadata 和调度开销。可以按 chunk 或 block 分组，把细粒度动态性约束在可编译范围内；也可以在高风险任务回退到 dense 或 retrieval。

线上 trace 应记录 pattern、非零 block 数、fallback、mask 生成时间、TTFT 和质量。没有这些信息，无法判断收益来自稀疏算法还是请求分布偶然变化。

## 26.18 稀疏模式的任务依赖

同一个 block pattern 对局部分类、长距复制和多证据问答的表现可能完全不同。局部任务只要求邻近信息，随机边和 global token 可能没有收益；精确引用要求关键证据有稳定、低噪声的路径；多证据任务还要求多个远程节点在同一 query 下汇合。

因此模式搜索不能只优化平均 loss。应把证据位置、实体数量、冲突版本、chunk 边界和工具参数作为独立维度，并报告每种失败是否由不可达、路径过长或读取错误造成。

## 26.19 Block size 是算法和硬件的共同参数

块稀疏把 token 连接量化为 block。block 越大，索引和 kernel 更规整，GPU 利用率可能更高，但无关 token 会一起被计算，稀疏收益下降；block 越小，连接更精确，却增加 metadata、调度和不规则访存。选择 block size 不能只看理论非零比例。

设理论非零比例为 `s`，块填充造成的额外计算比例为 `p`，稀疏 kernel 的有效利用率为 `u`，可以用一个教学近似估算实际收益：

```math
S_{\mathrm{real}}
\approx\frac{1}{(1-s)(1+p)/u+T_{\mathrm{index}}/T_{\mathrm{dense}}}
```

这不是硬件性能模型，但能提醒我们：即使 `s` 很小，block padding 和索引成本也可能吞掉收益。实际 benchmark 需要按长度、batch、pattern、dtype 和 kernel fallback 分桶。

## 26.20 动态稀疏的训练—服务契约

输入依赖稀疏会让每个请求拥有不同连接图。训练时要决定 pattern 是由规则、轻量 router 还是主模型生成；推理时要决定 router 的延迟、缓存、回滚和权限。若训练允许某个 token 访问全局证据，服务端的 mask 不能因为 block 对齐把它剪掉。

动态 pattern 还会影响 prefix cache。相同文本不一定产生相同的路由图，cache key 至少要包含 pattern/ router revision；如果路由结果不能稳定重放，就不应把动态稀疏的中间状态跨请求共享。

## 26.21 稀疏路径的能力验收

验收同时覆盖 local copy、长距 copy、文档引用、代码依赖、冲突版本和不可达证据。对每个任务记录有效路径长度、global token 使用、attention 非零数量、最终正确性和证据来源。还要把稠密 fallback 作为一种合法路径单独计费和计时。

在生产中，优先保证 mask 正确、输出协议正确和高风险任务可回退，再优化非零比例。稀疏度是实现参数，不是能力保证；一个连接更少但漏掉关键证据的 pattern，不能因为显存更小就被称为更优。

## 26.22 Block size 的选择例子

设序列长度为 8192，理想 token 级 pattern 只保留 10% 的连接。若 block size 为 16，连接边界较精细，metadata 较多；若 block size 为 128，连接需要向上填充，实际执行的非零 block 可能远高于理想 token 数。一个简单的实验是对同一 pattern 计算 token-level edge count、block-level edge count 和 padding ratio，再将它们送进同一个 kernel。

如果 block size 增大后理论非零比例不变但真实 kernel time 上升，原因可能是 padding；如果 block size 减小后理论 FLOPs 下降但 time 上升，原因可能是索引和 launch；如果质量变化只发生在块边界，说明 block 化改变了原本的证据路径。块大小是算法、编译器、硬件和任务共同决定的参数，不能从稀疏比例单独推导。

对动态 sparse，还要固定 pattern seed、router revision 和 prefix cache key。相同文本如果产生不同 block 图，就不能复用同一个中间状态；不同文本若意外共享 page，则可能造成信息泄露。质量、性能和隔离测试必须一起做。

## 26.23 稀疏模式的选择条件

Longformer（https://arxiv.org/abs/2004.05150）和 BigBird（https://arxiv.org/abs/2007.14062）展示了局部、全局和随机连接的代表性设计；Reformer（https://arxiv.org/abs/2001.04451）展示了 LSH attention 的另一条路线。

稀疏 attention 的难点不是“把矩阵变稀疏”，而是保证信息路径、mask 正确、kernel 高效和质量退化可控。它是一套算法与系统共同设计的问题。

## 26.24 Token 稀疏、块稀疏与结构化稀疏不是一回事

token-level sparse mask 可以精确地删除某些 query-key 连接，但 GPU 通常更喜欢规则的矩阵块。block-sparse 把连接按 `b\times b` 区域组织，硬件执行更规整，却会把块内本来不需要的 token 一起算进去。结构化稀疏还可能进一步限制每行、每列或每个 head 的非零模式，以换取编译和缓存的稳定性。

设理想 token 级非零边数为 `E_token`，块化后实际执行的边数为 `E_block`，则块填充开销为：

```math
\rho_{\mathrm{pad}}=\frac{E_{\mathrm{block}}-E_{\mathrm{token}}}
{\max(1,E_{\mathrm{token}})}.
```

`\rho_pad` 越高，理论稀疏率越可能与真实收益脱节。实验报告要同时给出三种数字：mask 中的非零 token 比例、执行的非零 block 比例和 kernel 实测 FLOPs/带宽。只报第一种数字，会高估块稀疏效果。

## 26.25 Sparse mask 的正确性测试

稀疏 mask 的单元测试不应只验证输出 shape。可以把每个 token 的输入替换为唯一的 basis 标记，运行一层 attention 后检查：允许的位置是否能影响输出，禁止的位置是否严格为零；再用 causal、padding、global 和跨 chunk 组合测试边界。

对于近似或 fused kernel，直接检查 dense attention 权重可能不可行，但仍可以做差分断言：把一个禁止可见的 token 改成极大值，目标位置的输出不应变化；把一个允许可见的 token 改变，输出应在数值容差内变化。测试要覆盖不同 batch、序列长度、block size、dtype、padding 和 page layout，避免只在偶数长度上通过。

服务端还要检查 mask 版本是否进入 cache key。若同一 prefix 先用 dense path、后用 local path，却复用了相同 KV page，质量问题和跨请求信息混用都会变得难以定位。pattern、position、dtype、模型 revision 和 cache schema 应在 artifact manifest 中绑定。

## 26.26 动态稀疏的 router 成本与错误归因

输入依赖的 sparse pattern 需要一个规则或 router 决定哪些连接保留。router 的计算、通信和决策延迟不应被隐藏在“attention FLOPs 减少”之后。若 router 选错了关键证据，错误发生在 pattern 生成；若证据可达但模型没有读取，错误发生在表示或训练；若 kernel fallback，错误发生在执行系统。三者修复方式完全不同。

可以把一次请求的耗时拆成：

```math
T_{\mathrm{total}}
=T_{\mathrm{route}}+T_{\mathrm{index}}+T_{\mathrm{kernel}}
 +T_{\mathrm{queue}}+T_{\mathrm{fallback}}.
```

线上 trace 至少记录 pattern revision、非零 block 数、router confidence、kernel path、fallback reason 和证据任务结果。没有这些字段，团队很容易用“增加模型规模”去修复一个本应由 router 或 kernel 解决的问题。

## 26.27 稀疏方案的最终比较表

一个可迁移的对比实验可以按下面的维度组织：

| 维度 | dense baseline | token sparse | block sparse | dynamic sparse |
| --- | --- | --- | --- | --- |
| 语义 | 直接全连接 | 精确删边 | 块内近似保留 | 输入依赖 |
| 主要成本 | `T^2` 计算/IO | 索引和不规则访存 | padding 和 kernel 约束 | router、索引和回退 |
| 主要风险 | 长度和 KV | 可达性不足 | 边界路径改变 | 路由不稳定 |
| 必测指标 | 质量、TTFT、显存 | 非零边、fallback | 非零 block、padding | router、p99、回退 |

四种路径必须使用相同 tokenizer、训练 token、位置方案、batch、dtype 和输出协议。质量不能只看总体 loss，还要看长距复制、冲突实体、多证据引用、工具 JSON 和不存在证据时的拒答。性能不能只看平均 throughput，还要看 p99、峰值显存、调度等待和单位成功成本。

稀疏 attention 的成熟标志，是系统可以回答“哪些边被删掉、为什么删、关键证据如何到达、实际用了什么 kernel、失败时如何恢复”。如果只能回答“理论复杂度更低”，那仍然只是论文摘要，不是可部署知识。
