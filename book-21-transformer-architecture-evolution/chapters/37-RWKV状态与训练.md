# 第三十七章：RWKV：RNN 式状态与 Transformer 式训练

## 37.1 RWKV 的核心目标

RWKV 试图把两种能力结合：训练时使用类似 Transformer 的并行计算，推理时使用类似 RNN 的固定状态。它不保存每个历史 token 的完整 K/V，而是维护按时间衰减的加权统计，当前 token 通过 time-mix 和 channel-mix 读取/更新状态。

这个方向的重要性不在于一个缩写，而在于它提出了一个系统问题：能否把长序列的训练并行和在线推理状态同时设计，而不接受 full attention 的 cache 增长。

## 37.2 小白直觉：带遗忘速度的加权账本

普通 RNN 把历史压成一个状态；RWKV 的 time-mix 更像多个带不同衰减速度的账本。最近的 token 权重较大，较早 token 逐渐衰减；当前 query 的内容又决定从账本中读取什么。

如果某个 token 很重要，模型需要把它写进合适的通道并让衰减足够慢；如果它过时，模型需要让旧贡献消失。固定衰减与内容门控共同决定记忆。

## 37.3 WKV 的教学抽象

可以把一个通道的历史统计写成：

~~~math
N_t=\sum_{i\le t}e^{-(t-i)w}k_i v_i,\qquad
D_t=\sum_{i\le t}e^{-(t-i)w}k_i
~~~

输出近似为：

~~~math
y_t=\frac{N_t+u\,k_t v_t}{D_t+u\,k_t}
~~~

其中 `w` 控制衰减，`u` 是当前 token 的 bonus/直接通路。实际 RWKV 版本对 time-mix、key/value/receptance 和数值稳定有具体参数化；上式只解释“衰减加权统计”。

## 37.4 Time-mix 和 Channel-mix

time-mix 处理跨时间信息，channel-mix 提供 token 内部的非线性变换。一个教学化 time-mix 可以写成：

~~~math
\tilde x_t=\mu_t\odot x_t+(1-\mu_t)\odot x_{t-1}
~~~

receptance gate 再控制输出：

~~~math
y_t=\sigma(r_t)\odot \operatorname{WKV}(k_t,v_t)
~~~

channel-mix 类似 gated FFN，通过不同投影产生 key/value 和 gate。具体版本的 layer norm、residual 和 token shift 需要查对应实现。

## 37.5 并行训练和递归推理

训练时，所有位置的 time-mix 可以通过并行 prefix/scan 或核化形式计算；推理时只保存每层的统计 state。状态大小不随历史长度增长：

~~~math
M_{\mathrm{state}}\approx L\cdot H\cdot d_{\mathrm{state}}\cdot b
~~~

但 state 不是可随机访问的原文。对长流服务，固定 state 能减少内存；对多证据引用，可能需要外部检索或 global path。

## 37.6 一个教学版衰减状态

~~~python
import math


def rwkv_like(values, decay, bonus=0.0):
    numerator = 0.0
    denominator = 0.0
    outputs = []
    for key, value in values:
        numerator = math.exp(-decay) * numerator + key * value
        denominator = math.exp(-decay) * denominator + key
        outputs.append((numerator + bonus * key * value)
                       / (denominator + bonus * key + 1e-8))
    return outputs


print(rwkv_like([(1.0, 2.0), (0.2, 8.0), (1.0, 3.0)], decay=0.4))
~~~

这个代码没有 gate、向量通道和稳定的 log-space 计算，只帮助观察 decay 和 bonus 对近期/当前信息的影响。

## 37.7 内容记忆的边界

RWKV 的衰减统计可以保留趋势和近期事实，但历史 token 被汇总。若两个实体的 key/value 统计相近，后续 query 可能不能恢复区别。长程精确复制、变量绑定和冲突版本是关键评测。

适合的任务包括流式生成、持续对话、低内存边缘推理和顺序信号；需要任意文档跳转的任务要加入 retrieval 或 attention 补偿。

## 37.8 与 Transformer 的 trade-off

Transformer 的 attention score 随 query 动态访问显式历史，ICL 和多证据组合更自然；RWKV 的固定 state 让 decode 内存稳定，训练/推理路径统一得更像 RNN。两者都需要成熟 kernel、量化和 serving。

RWKV 不应被描述成“完全没有 attention 的 Transformer”。它保留了 time-mix/channel-mix 的设计和对并行训练的追求，信息流是递归统计而非 full token attention。

## 37.9 常见失败模式

包括 decay 参数导致 state 消失或爆炸、log-space 稳定性错误、time-mix 位置错位、padding 更新状态、训练并行与递归推理不一致、state 迁移丢层、以及将模型版本的 WKV 实现混写。

排查先在短序列对比逐步 reference，再测不同长度和 dtype；保存每层 state checksum；做 prefix continuation、batch reorder、preemption 和跨请求隔离。

## 37.10 机制与边界：状态统计与内容寻址的折中

RWKV 类更新通常把过去压成加权 key-value 统计，提供比单向量 state 更丰富的内容相关读写；但统计容量仍由通道/矩阵维度决定。衰减参数提供时间尺度先验，receptance 提供输出门控，两者在训练中共同形成记忆策略。

论文和实现比较要关注：time-mix 的具体公式、WKV 的数值表示、训练并行算法、state shape、量化支持和长程任务，而不是只看“RNN-like”标签。

## 37.11 面试追问、误区与练习

**问：RWKV 如何兼顾训练并行和推理递归？**

标准回答：它把跨时间计算写成可并行的加权统计/scan 形式用于训练，同时在推理时维护等价或近似的递归 state；状态固定但不保留所有历史 token。

**问：RWKV 的主要代价是什么？**

标准回答：精确任意历史检索和复杂 ICL 可能受状态压缩影响；kernel、状态恢复、长程评测和生态也需要单独验证。

常见误区包括把 time-mix 当滑窗、把 state 当 KV、把所有版本用同一 WKV 公式描述，以及忽略数值稳定。

练习：实现不同 decay 的加权 state，加入 STORE/RESET/冲突版本任务，比较精确 recall 和 state 大小。

## 37.12 衰减参数和记忆时间尺度

RWKV 类路径的时间混合依赖不同的衰减速度。快衰减维度关注近期 token，慢衰减维度保留更长趋势；如果所有维度的时间尺度相近，模型要么忘得太快，要么被旧信息污染。

训练时可以统计不同 channel 的有效 half-life，并在局部复制、长程复制和冲突版本任务上做相关性分析。half-life 长不一定代表事实保留好，因为 state 还要承载内容和 query 对齐。

## 37.13 RWKV 与显式检索的边界

递归 WKV 可以低成本处理持续流，但当前 query 不能任意回看每个历史 token。需要精确引用时，可以保留外部索引或少数显式 attention 层。一个实用系统不是把两条路径互斥，而是让 RWKV 负责扫描、retrieval 负责定位。

## 37.14 训练形态与递推形态的等价性

RWKV 类模型常希望训练时利用并行矩阵运算，推理时只保留递归状态。两种路径必须有相同的时间混合、初始化和边界语义；如果训练使用完整序列而推理每段 reset，模型看到的历史分布就不同。

用同一 prefix 做 parallel、chunk 和 recurrent 对照，保存每一步 hidden、logits、state norm 和最终 token。差异若随长度增长，优先检查衰减、累计精度、padding 和 chunk boundary。

## 37.15 适用任务和回退策略

RWKV 适合连续文本流、固定状态预测和资源受限的 decode；需要任意远程引用或多版本冲突时，保留 retrieval 或 global attention。路由器要按任务风险选择路径，并在引用验证失败时回读原文。

## 37.16 用半衰期和容量理解 RWKV 的状态

对某个时间混合通道，若衰减因子近似为 a，旧信息影响可用 a^j 描述。半衰期为：

~~~math
j_{1/2}=\frac{\log 0.5}{\log a}
~~~

但 RWKV 的 state 不只保存一条标量衰减曲线。不同 channel、key-value 统计和 receptance 共同决定最终可读信息。一个 channel 的半衰期很长，只表示它不容易忘记，不表示它能区分两个相似实体；多个实体写入同一统计量时仍会发生干扰。

可以构造四类最小任务来区分这些能力：短距复制、远距复制、两个实体冲突、无目标时的误报。分别改变序列长度、实体数量和 state 维度，记录 exact match、false positive、state norm 和每 token 更新时间。这样能够回答“RWKV 记忆很长吗”之外更具体的问题：它记住的是什么，能否被正确读取，代价是多少。

## 37.17 WKV 的数值稳定与实现版本

带指数权重的累计在长序列上容易出现下溢、溢出或大数相除。工程实现往往使用 log-space、分块归一化、重标度或专门 kernel；这些细节会影响并行训练和递归推理的一致性。不同 RWKV 版本在 time-mix、channel-mix、WKV 表达和 state shape 上并不完全相同。

因此复现论文公式时不能只抄一个递推式。应先写一个高精度 reference，再与优化 kernel 对照：

~~~math
\epsilon_t=\lVert s_t^{(\mathrm{ref})}
-s_t^{(\mathrm{kernel})}\rVert_\infty
~~~

检查 ε_t 是否随长度增长，检查 BF16/FP16 是否比 FP32 更早失稳，并把 padding、reset 和 chunk 边界纳入测试。若模型输出变化只发生在很长序列，短样本的单元测试并不能证明服务安全。

## 37.18 状态服务的生命周期

RWKV 的固定状态降低了按 token 保存 KV 的压力，却引入了另一类服务契约。状态必须与请求、模型 revision、tokenizer、dtype 和 batch slot 绑定；请求结束后要销毁或显式交还，不能依赖调用方“记得清空”。抢占和迁移时需要保存每层 state，而不只是最后一个 token 的位置。

推测解码也会放大这个问题。若草稿模型先写入 RWKV state，主模型拒绝部分候选，系统必须回滚到接受前的 state；只回滚 token 文本而不回滚状态，会让被拒绝 token 继续影响后续输出。这个约束说明固定状态并不等于无状态，反而要求 serving engine 明确管理状态版本。

## 37.19 WKV 的数值和状态语义

RWKV 类模型用带衰减的加权状态表达历史，训练时希望并行计算，推理时按递归状态更新。指数权重、归一化和有限精度决定了旧 token 的影响是否稳定；长序列中应检查分子/分母的尺度、溢出、下溢和空状态初始化。

验证时比较 full parallel、chunked 和 step recurrence 的 logits，并加入不同长度、padding、reset、batch reorder 和 state restore。只在短序列上通过，不代表在线长流中的状态语义正确。

## 37.20 RWKV 的服务状态契约

每个请求需要保存 time-mix/channel-mix 所需的 state、逻辑位置、模型 revision、adapter、dtype 和 reset 标记。状态不能跨租户共享，也不能因为 prefix 文本相同就直接复用；不同系统提示或权重版本可能改变隐状态。

与 KV cache 相比，RWKV 状态更小但更难回读原文。高风险引用任务仍需要外部 artifact 或检索；状态快照只能说明模型当时的内部表示，不能替代来源、权限和删除记录。

## 37.21 WKV 状态的数值稳定与服务检查

一个简化的递归表示为：

```math
a_t=\lambda_t a_{t-1}+\exp(k_t)v_t,
\qquad
b_t=\lambda_t b_{t-1}+\exp(k_t),
\qquad
y_t=\frac{a_t}{b_t}.
```

真实实现会使用 log-sum-exp、分组状态或更稳定的重参数化。服务端不能直接把 `a_t`、`b_t` 当作跨版本可读的普通缓存；dtype、decay、key scaling、position 和 reset 都属于 state schema。长流压测要检查状态范数、NaN、chunk continuation 和从 snapshot 恢复后的 token 一致性。

## 37.22 用状态容量解释“记得久”

递归模型的状态大小固定，并不代表可无损记住的事实数量固定不变。状态中的通道、key-value 统计、衰减时间尺度和输出门控共同决定可分辨的信息。可以把状态容量写成教学化的约束：

```math
I_{\mathrm{usable}}
\le f(d_{\mathrm{state}},H,\{j_{1/2,h}\},\mathrm{precision},\mathrm{training})
```

这个函数没有通用闭式解，但它提醒我们不要只报告“每请求 state 只有多少字节”。如果多个实体共享同一统计量，状态仍可能发生冲突；如果所有 channel 都快速衰减，远程信息会消失；如果慢衰减 channel 太多，旧信息又会污染新 query。

最小能力实验应同时包含远距复制、两个实体冲突、版本更新和无目标误报。比较 exact recall、冲突选择率、state norm、每 token 更新时间和 snapshot 大小，才能知道固定状态节省了什么，又牺牲了什么。

## 37.23 训练课程与在线分布必须一致

训练阶段可以使用并行 scan，在线阶段却按 token 更新 state。若训练样本经常在短段边界 reset，而线上请求持续数十万 token，模型会面对不同的状态分布；反过来，训练中总是保留完整历史，线上频繁抢占和恢复，也可能造成状态断裂。

训练回归要加入 chunk continuation、padding、reset、state restore 和 batch reorder。每个实验保存 prefix 后的 state checksum 和下一 token logits，比较 parallel、chunked、recurrent 三条路径。若差异只在长序列出现，优先排查累计精度和边界语义，而不是先增加模型层数。

## 37.24 一个在线状态事故

某服务为了复用 batch slot，在请求 A 结束后把 slot 交给请求 B，但只清除了 token offset，没有清理每层 WKV state。B 的第一个回答仍然流畅，却偶尔引用 A 的实体。这个事故很难通过普通质量均值发现，因为只有相似主题或特定长历史才会触发。

正确的生命周期是：请求绑定 state owner 和 model revision，完成/取消时原子释放，slot 复用前做零状态或 checksum 检查，snapshot/restore 必须带 schema、dtype、position 和租户。状态比文本更隐蔽，不能依赖调用方“记得 reset”。

## 37.25 证据范围与路线选择

RWKV 的公开论文和实现支持其并行训练、递归推理和 WKV 路线；不同版本的 time-mix、WKV 数值形式、state shape 和 kernel 不能互换。本文用加权 key-value 统计解释机制，不把教学递推式当作所有版本的完整实现。

工程选择应按查询形态决定：持续流和固定状态任务可以优先考虑递归路线；需要任意远程精确引用时，应加入外部检索、少量 global attention 或原文回读。架构名称不能替代任务评测和状态服务演练。

## 37.26 WKV 数值路径与状态隔离

RWKV 的 WKV 计算看似只是加权历史，但长序列上会遇到指数衰减、累计尺度和低精度问题。实现通常需要重标度或稳定的递推形式；验证时应比较 reference、并行 scan 和单步路径在不同长度、dtype、衰减参数下的输出误差。

服务侧还要把 WKV state 当作租户隔离对象。请求结束、取消、超时和 batch slot 复用都必须原子清理；snapshot 要带模型 revision、time-mix 配置、position/step 和 checksum。一个主题相关的错误回答可能不是模型能力问题，而是旧请求 state 没有清干净。

## 37.27 小结与资料边界

RWKV 可参考 *RWKV: Reinventing RNNs for the Transformer Era*（https://arxiv.org/abs/2305.13048）。不同 RWKV 版本的 time-mix/WKV 细节不同，本文采用教学抽象。

RWKV 的价值是把并行训练和固定推理状态放在一条路线中。它证明了 attention 之外存在有工程意义的序列信息流，但不自动解决所有长程检索问题。
