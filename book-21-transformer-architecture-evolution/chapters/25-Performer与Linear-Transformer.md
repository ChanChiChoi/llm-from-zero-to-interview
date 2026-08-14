# 第二十五章：Performer、Linear Transformer 与注意力近似

## 25.1 两条看似相同、实际不同的路线

Linear Transformer 主要利用可结合的特征形式重排因果 attention，把历史累积成状态；Performer 则重点研究如何用随机特征近似 softmax kernel，从而避免显式构造二次矩阵。两者都可能出现线性序列成本，但一个更接近结构重写，一个更接近核近似。

把它们都简称为“线性 attention”会忽略两个问题：近似误差从哪里来，以及模型是否仍然拥有足够的内容寻址能力。对于短序列和平均 loss，差异可能不明显；对于长序列的 top evidence 排序，差异可能放大。

## 25.2 Softmax kernel 的困难

标准 attention 的未归一化相似度是：

~~~math
K(q,k)=\exp(q^\top k)
~~~

若能找到有限维特征映射 `\phi` 使：

~~~math
K(q,k)\approx\phi(q)^\top\phi(k)
~~~

就可以把 key/value 先聚合。但是指数内积的精确特征维度通常不可控，有限随机特征只能在统计意义上近似。近似误差受到特征数、采样方式、query/key 范围和温度影响。

## 25.3 Random feature 的教学形式

随机 Fourier feature 的一个教学形式是：

~~~math
\phi(x)=\sqrt{\frac{2}{m}}\cos(Wx+b)
~~~

其中 `W` 是随机矩阵，`b` 是随机相位。不同 kernel 需要不同采样分布；softmax kernel 还需要正值、数值稳定和归一化设计，不能直接把任意 RFF 代入 Transformer 就得到 Performer。

一个简单的随机特征代码如下：

~~~python
import torch


class RandomFeatureMap(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("weight", torch.randn(feature_dim, input_dim, generator=generator))
        self.register_buffer("phase", 2 * torch.pi * torch.rand(feature_dim, generator=generator))

    def forward(self, x):
        # x: [batch, seq, input_dim], output: [batch, seq, feature_dim]
        z = torch.einsum("bsd,fd->bsf", x, self.weight) + self.phase
        return (2.0 / self.weight.size(0)) ** 0.5 * z.cos()


features = RandomFeatureMap(8, 32)
print(features(torch.randn(2, 5, 8)).shape)
~~~

这个映射只用于讲解随机近似，不等于 Performer 的 FAVOR+ 实现。生产版本会处理正值特征、正交随机矩阵、稳定归一化和 causal/non-causal 两种路径。

## 25.4 FAVOR+ 的核心直觉

Performer 通过正值随机特征近似 softmax attention，使分子和分母都能按结合律计算。抽象地写：

~~~math
\operatorname{Attn}(Q,K,V)
\approx
\frac{\phi(Q)(\phi(K)^\top V)}
{\phi(Q)(\phi(K)^\top\mathbf{1})+\epsilon}
~~~

非因果场景可以先在整段上聚合；因果场景则按位置递归更新。特征数量 `m` 越大，近似方差通常越低，但状态和计算也越大。低方差并不保证任务级精确检索完全不受影响。

## 25.5 Linear Transformer 的精确重排

如果 attention 形式已经是 `\phi(q)^T\phi(k)`，则：

~~~math
\phi(Q)\left(\phi(K)^\top V\right)
\quad\text{可以先计算}\quad
S=\phi(K)^\top V
~~~

因果情况下：

~~~math
S_t=S_{t-1}+\phi(k_t)v_t^\top
~~~

这一步是代数上的重排，不是随机近似；但它改变了 softmax 的形式，并带来固定状态的信息瓶颈。Linear Transformer 论文展示了如何用这一形式进行快速自回归建模，实际变体仍需单独分析。

## 25.6 误差如何传到最终答案

设理想输出为 `y`，近似输出为 `\hat y`。即使每个 score 的均方误差小，softmax 后的 top 权重排序也可能改变：

~~~math
\Delta_{\mathrm{rank}}=
\mathbb{1}[\operatorname{argmax}_i a_i
\ne \operatorname{argmax}_i\hat a_i]
~~~

如果任务依赖一个稀有 token，top-1 排序变化会导致答案完全错误；如果任务是主题分类，多个近似误差可能互相平均。评估必须按任务结构测，而不能只报告平均 kernel error。

## 25.7 计算和内存的三方取舍

设特征维度为 `m`、head 维度为 `d`。线性状态约为 `m×d`。当 `m` 接近序列长度或 head 很多时，优势缩小；当 `m` 很小，近似误差增大。一个粗略成本模型为：

~~~math
C_{\mathrm{linear}}\approx Tmd+Tmd_v,\qquad
M_{\mathrm{state}}\approx m d_v
~~~

还要加 projection、normalizer、kernel launch 和状态读写。GPU 对大矩阵 dense GEMM 很友好，而细小递归更新可能没有理论复杂度那么快。

## 25.8 与 Performer、FlashAttention 的区别

Performer 改变数学计算，通过特征映射近似 softmax；FlashAttention 保持 attention 数学形式，改变内存访问和 kernel 组织。前者有近似误差，后者通常没有模型层面的近似误差；前者可能减少长序列的算术复杂度，后者主要减少中间存储和带宽。

两者可以分别评测：固定模型质量比较 Performer 与 dense；固定数学 attention 比较 FlashAttention 与 naive kernel。把二者的 benchmark 直接放在一条曲线里，会混淆模型误差和实现速度。

## 25.9 失败模式和排查

常见问题包括特征维度不足、随机种子造成结果漂移、指数/正值映射溢出、分母过小、长序列累计误差、padding 参与状态、训练时非因果而推理时因果、以及 checkpoint 与 feature map 配置不匹配。

排查顺序：先在固定 Q/K/V 上比较 dense 参考和近似输出；检查分母、范数和 top-k 排名；再做不同 feature 数和随机 seed；然后按长度、距离和稀有 token 分桶；最后 profile kernel。不要先从平均 perplexity 下结论。

## 25.10 机制与边界：随机特征的方差和任务分布

随机特征近似通常给出期望或高概率误差界，但界限依赖输入范数、特征数和采样分布。语言模型的 query/key 分布随层、token、位置和训练阶段变化，固定随机映射未必在所有区域都均匀准确。正交特征、reweighting、kernel temperature 和训练共同决定实际效果。

此外，近似 attention 可能改变梯度噪声和优化景观。一个在 inference 上看似可接受的近似，不一定在 pretraining 中稳定。要区分从头训练、蒸馏替换、推理近似和 kernel-only 优化四种实验。

## 25.11 面试追问、误区与练习

**问：Performer 和 FlashAttention 的主要差别是什么？**

标准回答：Performer 通过随机特征近似 softmax kernel，改变计算形式并引入近似误差；FlashAttention 保持精确 attention，利用 tiling 和 online softmax 降低中间矩阵访存。

**问：增加随机特征数是否一定更好？**

标准回答：通常能降低近似方差，但会增加 state、计算和带宽；还要看 kernel、长序列任务和硬件，不能只按特征数排序。

常见误区包括把所有线性 attention 当作 Performer、把近似误差等同于平均 loss 误差，以及用小随机 seed 实验声称通用优势。

练习：实现 dense、Linear Transformer 和随机特征 attention，在不同 feature_dim、序列长度和稀有 token 检索任务上比较误差/吞吐曲线。

## 25.12 近似误差如何进入 softmax

Performer 用随机特征近似 softmax kernel 时，误差不仅发生在 score，还会经过归一化、value 聚合和后续层。若分母估计偏小，输出可能被放大；若随机特征方差大，长序列的误差会累积。

因此随机特征数量 m 是质量和计算的旋钮。固定 seed 的单次结果不能代表平均性能，应比较多个 seed、不同 m、不同长度和不同任务，并报告方差而不只是均值。

## 25.13 近似 attention 的回退策略

高风险精确引用可以走 full attention 或 retrieval，普通流式摘要可以走近似路径。路由需要基于任务风险和证据需求，而不是只基于序列长度。

线上应记录 approximate path、feature seed、fallback reason 和最终质量。没有这些字段，模型在一次请求中从近似路径切换到 full path 后，无法解释延迟和答案变化。

## 25.14 随机特征的方差预算

Performer 类方法用有限随机特征近似核函数。设特征数为 m，随机估计的误差通常会随 m 增大而下降，但具体方差还取决于输入分布、核值范围、归一化和序列长度。不能把误差写成只由 m 决定的固定常数。

实验应使用多个随机 seed，并报告均值、标准差和最差分位数。对高风险字段，最差分位数比平均 loss 更重要，因为一次漏掉关键证据就可能使整个任务失败。

## 25.15 近似路径的数值保护

随机特征输出可能出现极端值，分母归一化、指数和累积精度都要有保护。实现中要检查 feature map 的 dtype、是否在 log-space 重标度、分母下限以及长度增加时的 state norm。

可以将 full attention 作为 reference，在相同 Q/K/V 上比较：

~~~math
E_{\mathrm{out}}
=\frac{\lVert Y_{\mathrm{approx}}-Y_{\mathrm{full}}\rVert_F}
{\lVert Y_{\mathrm{full}}\rVert_F+\varepsilon}
~~~

再把张量误差映射到 exact recall 和引用支持，避免把较小的平均误差误判成没有业务影响。

## 25.16 何时使用 Performer，何时使用 FlashAttention

Performer 改变了数学近似和状态形式，FlashAttention 主要改变精确 attention 的 IO 组织。二者不能只按“都是高效 attention”比较：前者可能牺牲精确性换取更稳定的历史存储，后者保留 full attention 语义但仍受 KV 和长度约束。

如果任务要求任意精确引用、上下文长度在成熟 kernel 能承受的范围内，精确路径可能更稳；如果输入是持续长流且允许压缩，核化路径才有明显动机。最终应以 matched workload 的质量、资源和回退成本决定。

## 25.17 随机特征的误差如何进入模型输出

若用随机特征 `\phi` 近似 softmax kernel，单个 pair 的估计误差可能很小，但 attention 是对大量 token 的加权归一化，误差会通过分子和分母共同传播。重要的不是平均 kernel MSE，而是查询相关的 attention rank、top-k 证据保留率和最终任务错误。

可以对同一批 Q/K 使用不同随机种子，测输出方差：

```math
\mathrm{Var}_{s}[y_t]
=\frac{1}{S}\sum_{s=1}^{S}
\left(y_t^{(s)}-\bar y_t\right)^2
```

如果方差在长序列、尖锐分布或数字/代码 token 附近显著升高，说明近似路径不能只用平均吞吐决定。增加特征数可能降低方差，却会增大状态、带宽和 kernel 成本；固定长度的随机 seed 也不能代替跨模型版本的回归。

## 25.18 什么时候需要稠密回退

近似 attention 不必在所有 token 上使用同一策略。系统可以对短序列、低风险或平滑任务走 linear path，对需要精确检索、结构化输出和高置信引用的局部窗口走标准 attention；也可以在发现分母异常、候选分歧或近似方差超阈值时回退。

回退要定义状态语义：如果 linear path 已经更新了递归状态，切换到稠密路径时需要从可靠 checkpoint 重算，不能把两个路径的 hidden 直接拼接。回退事件、原因、重算成本和最终质量必须进入 trace，否则线上只会看到“偶发变慢”。

## 25.19 Performer、Linear Transformer 与 FlashAttention 的实验边界

Performer 类方法改变 attention 的近似计算，Linear Transformer 通过重排得到递归状态，FlashAttention 则尽量精确地改变内存访问。公平比较不能只比较复杂度或单 kernel tokens/s；应固定质量验收条件、上下文长度、batch、dtype、硬件和输出协议，报告误差、显存、p99、回退比例和单位成功成本。

公开论文能支持方法的数学抽象和实验条件，但不能推断某个新 runtime 对所有模型都自动使用随机特征或线性路径。具体实现的 feature map、归一化、因果语义和数值保护都要查代码和做 golden replay。

## 25.20 随机特征数如何进入系统决策

设随机特征数为 `m`。增加 `m` 通常能降低核估计方差，却会增加 projection、状态和内存流量；减少 `m` 可能提高吞吐，却可能丢失尖锐的相似度边。不要把 `m` 当成越大越好的单调旋钮，而要在任务验收条件下寻找可接受区间。

一个可复现实验可以固定同一组 Q/K/V，设置 `m` 为 64、128、256、512，使用 5 个随机 seed。对每个设置同时报告 full attention 的输出误差、top-k evidence retention、数字 exact recall、state bytes、kernel time、p99 和 fallback rate。若 `m=64` 的平均误差很小，但某个 seed 在冲突版本任务上完全漏掉关键证据，那么生产验收条件应看最差分位数，而不是均值。

这也解释了为什么 Performer 和 FlashAttention 不能只比较论文中的复杂度。前者改变了数学近似和状态容量，后者尽量保留 dense attention 的语义而优化 IO。若系统可以在高风险请求上回退到 dense path，必须把回退率和重算成本计入单位成功成本；如果回退频繁，近似路径可能只是增加了复杂度。

## 25.21 线性化近似的适用条件

Linear Transformer 见 https://arxiv.org/abs/2006.16236，Performer 见 https://arxiv.org/abs/2009.14794。论文中的复杂度和误差结论不等于特定 GPU、模型规模和任务上的端到端结果。

Performer 代表“用特征近似减少 attention 交互”，Linear Transformer 代表“用结合律维护递归统计”。它们都值得学习，但选择时必须明确接受了哪一种近似和信息压缩。

## 25.22 因果线性注意力的递推状态

把特征映射记为 `\phi(q)` 和 `\phi(k)`，因果线性 attention 常把：

```math
y_t=\frac{\phi(q_t)^\top\left(\sum_{s\le t}\phi(k_s)v_s^\top\right)}
{\phi(q_t)^\top\left(\sum_{s\le t}\phi(k_s)\right)+\epsilon}.
```

定义矩阵状态 `S_t` 和归一化向量 `z_t`：

```math
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\qquad
z_t=z_{t-1}+\phi(k_t).
```

这就是它能在线性时间运行的原因：每一步只更新固定大小的 state，再用 `\phi(q_t)` 查询。代价是历史 token 不再以独立 K/V 的形式保留，更新后的 state 可能无法恢复一个具体的远程 token。对流式信号和局部平滑预测，这种压缩可能足够；对需要精确引用某个数字的任务，它就是主要风险。

## 25.23 归一化不是实现细节

线性 attention 的分母负责把不同长度和不同相似度尺度的输出放在可比较的范围。如果 `\phi` 产生负值、分母接近零或半精度累积溢出，模型可能出现异常放大、NaN 或长序列漂移。不同论文使用的 feature map、正值变换和数值保护不能直接互换。

最小实现应在训练和推理中统一 `epsilon`、累积 dtype、state reset 和 mask 语义。可以构造三个 sanity check：全零输入、重复相同 token、逐 token 推理与整段因果推理的等价性。若逐 token 与整段输出差异很大，通常不是“线性 attention 理论失效”，而是 state 更新顺序、归一化或位置处理不一致。

## 25.24 递归状态的容量与可寻址性

标准 KV cache 的历史大小随序列长度增长，但每个位置仍有独立向量；线性 attention 的 state 大小近似由 feature dimension `m`、value dimension `d_v` 和层数决定：

```math
M_{\mathrm{state}}
\approx L\,(m d_v+m)\,b.
```

这可以显著降低长流的显存增长，但它也改变了“记住”的含义。状态容量大不代表可寻址容量高：多个相似 key 可能写入相同方向，后写入内容可能覆盖或混合前文。评测要加入位置交换、重复实体、冲突版本和多证据任务，观察 state collision 是否导致固定类型的错误。

一个直观实验是让模型先读入 `N` 条带唯一编号的事实，再查询其中一条。逐步增加 `N`，记录 exact recall、引用支持和状态范数。若摘要质量稳定但精确召回下降，说明模型保留了主题统计，却没有保留足够的寻址细节；此时增加上下文长度不一定有帮助，可能需要外部检索或混合 dense layer。

## 25.25 线性路径的部署决策

Performer/Linear Transformer 的部署不能只根据 `O(T)` 选择。应先确定 workload：持续传感流、语音帧和在线日志更关心稳定状态与每步延迟；代码跨文件、合同引用和结构化工具调用更关心原子证据和协议正确性。后者通常需要 dense、稀疏或 retrieval 旁路来补偿状态压缩。

实际报告至少包含：state bytes/request、state update time、长序列数值漂移、exact recall、结构化合法率、p50/p99、fallback 和重算成本。若系统可以混合使用线性和 dense block，还要说明状态如何交接。直接把一个递归 state 传给 dense attention 不等于恢复原始 K/V，交接层需要训练或明确的摘要协议。

选择规则可以写成一个硬性条件：

```text
use_linear =
    streaming_workload
    and state_stability_ok
    and exact_evidence_requirement <= threshold
    and fallback_cost_acceptable
```

这不是产品配置的完整代码，而是把“线性复杂度很诱人”转化为可检查的决策条件。近似 attention 的工程价值，来自可控的误差与成本，而不是复杂度符号本身。
