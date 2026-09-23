# Attention Residuals：让网络沿深度选择历史表示

> 资料核验：Kimi Team，*Attention Residuals*，arXiv:2603.15031，2026-03-16；对应官方仓库与论文源码。本文先讲论文公开的机制，再讨论教学实现和工程边界。Attention Residuals 与 Kimi K3 的关系是：Kimi K3 官方发布文章明确提到采用 AttnRes，但没有公开 K3 的全部层数、块划分和训练配置。下文的实验数字来自 Attention Residuals 论文，不是 K3 的实测结果。

## 先从一个“不断改稿”的例子开始

假设你和同事一起写一份很长的技术方案。第一版提出了目标，第二版补充了系统边界，第三版加入了成本约束，第四版发现了一个安全漏洞。最简单的协作方式，是把每一版全文首尾相接。版本越来越多，文档会越来越长，后来的人也很难判断应该相信哪一版。

Transformer 的残差连接有一点相似：每个 block 都把自己的输出加回主干状态。第 1 层写入一点信息，第 2 层再写入一点，经过几十、几百层后，主干状态是很多历史输出的累加。它的优点是梯度有一条直接通路，缺点是每一层都用固定的“加法”处理历史：早期表示的重要性没有显式选择机制。

Attention Residuals（AttnRes）提出的想法很直接：让第 `l` 层像 Transformer 沿 token 维做注意力一样，沿“深度维”查看早先层的表示，然后决定当前层应该更多使用哪几层的结果。这里的注意力不是重新读取序列中的 token，而是读取网络深度方向的历史表示。这个坐标轴区别必须先分清，否则很容易把 AttnRes 误解成普通 self-attention 的另一个名字。

## 标准残差到底在做什么

把第 `i` 层的变换记为 `f_i`，输入状态记为 `h_i`。标准残差可以写成：

```math
h_{i+1}=h_i+f_i(h_i)
```

展开后：

```math
h_l=h_1+\sum_{i=1}^{l-1}f_i(h_i)
```

每个历史输出的系数都是 1。网络会通过训练改变 `f_i` 的内容和尺度，但残差路径本身没有一个“当前输入相关的选择权重”。随着深度变大，表示幅度、归一化和梯度分配需要额外控制；论文将这种固定单位权重的累积与 PreNorm 下的 hidden-state growth 和 contribution dilution 联系起来。

这并不意味着标准残差必然不稳定，也不意味着所有深层 Transformer 都需要 AttnRes。它说明了一个可研究的问题：如果历史表示的贡献可以按内容动态选择，是否能提高深度方向的信息流效率？

## Full AttnRes 的定义

论文把 token embedding 记作 `v_0=h_1`，前面各层的输出记作 `v_i=f_i(h_i)`。第 `l` 层从所有之前的表示中做加权和：

```math
h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}v_i,
```

并要求：

```math
\sum_{i=0}^{l-1}\alpha_{i\to l}=1,\qquad \alpha_{i\to l}\ge 0.
```

权重由 softmax 得到：

```math
\alpha_{i\to l}=\frac{\exp\left(w_l^\top\mathrm{RMSNorm}(k_i)\right)}{\sum_{j=0}^{l-1}\exp\left(w_l^\top\mathrm{RMSNorm}(k_j)\right)}.
```

在论文的设定中，`q_l=w_l` 是第 `l` 层独有的可学习伪查询向量，`k_i=v_i`。伪查询不依赖当前层的 token 输入，因此它不是普通 token attention 的 query；但 `k_i` 来自输入相关的中间表示，所以 softmax 权重仍会随输入变化。`RMSNorm` 的作用是避免某个历史输出仅仅因为幅度大，就压倒其他表示。

一个小例子：如果有三个候选表示，分数为 `[2, 1, 0]`，softmax 权重大约是 `[0.665, 0.245, 0.090]`。当前层主要使用第一个候选。如果输入改变导致分数变成 `[0, 1, 2]`，权重就会反过来。这就是“选择历史表示”与“把所有历史表示固定相加”的区别。

## 为什么要做 Block AttnRes

Full AttnRes 需要保存每一层的历史表示。论文把每个 token 的表示存储和深度访问开销概括为与层数 `L` 相关的量；在激活检查点、流水线并行和跨 stage 通信场景中，所有历史输出都保持可访问会变得昂贵。

Block AttnRes 把 `L` 层分成 `N` 个块。一个块内部仍然把各层输出累加成部分和，块与块之间才使用深度注意力。对第 `n` 个块的第 `i` 层，候选表示包括：token embedding、前面已经完成的块表示，以及当前块已经产生的部分和。于是：

```math
b_n=\sum_{j\in\mathcal B_n}f_j(h_j),
```

当前块的部分和记作 `b_n^i`。`N=L` 时，每层都可以看作一个块，接近 Full AttnRes；`N=1` 时只有一个块，块内退化为累加，但 embedding 与块内部分和之间仍有加权聚合；不能把它与原始标准残差逐元素等同。块数提供了表达能力和系统开销之间的旋钮。

论文源码还描述了推理时的两阶段计算：第一阶段把一个块中的伪查询批量对已完成块表示做 attention；第二阶段顺序处理当前块的部分和，再用 online softmax 合并两部分统计量。这样可以减少重复读取历史块，且保持数学上的 softmax 等价。流水线训练则通过 cross-stage caching 只传输新完成的块，减少重复通信。

## 一个零依赖的教学实现

下面代码刻意只实现 Full AttnRes 的核心权重计算，便于读者观察“沿深度维 attention”。它不包含论文中的并行 kernel、激活检查点、流水线缓存或 Block AttnRes 的生产优化。

```python
import math


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [x / z for x in exps]


def attention_over_depth(values, query):
    """values: list of [d] vectors; query: [d]."""
    keys = [[x / math.sqrt(sum(y * y for y in value) / len(value) + 1e-8)
             for x in value] for value in values]
    scores = [sum(q * k for q, k in zip(query, key)) for key in keys]
    weights = softmax(scores)
    output = [
        sum(weight * value[j] for weight, value in zip(weights, values))
        for j in range(len(values[0]))
    ]
    return weights, output


values = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
weights, output = attention_over_depth(values, [2.0, 0.0])
print("weights:", [round(x, 3) for x in weights])
print("sum(weights):", round(sum(weights), 6))
print("output:", [round(x, 3) for x in output])
```

运行时应满足 `sum(weights)=1`。把 query 改成 `[0.0, 2.0]`，权重会偏向第二个候选；把三个 value 同时乘以一个常数，在代码已有的 RMSNorm 下，权重不会只由绝对幅度决定。真实模型还需要处理 batch、sequence、head、参数初始化和反向传播，这段代码只用于建立坐标轴和加权聚合直觉。

## 工程取舍与边界

AttnRes 的主要收益是给深度方向增加内容相关的选择机制；主要成本是保存、读取和传输历史表示。Block 设计降低了存储和通信，但块内的普通累加仍然可能稀释细粒度信息。论文报告在其特定实验中约 8 个块能保留大部分收益，并报告流水线训练开销低于 4%、推理延迟开销低于 2%；这些数字依赖论文的模型、硬件和实现，不能当作所有 serving engine 的保证。

AttnRes 也不等于“记忆无限”。它仍然只访问有限层表示，softmax 仍会产生竞争，历史信息可能被压低；如果任务需要跨很长 token 范围检索，仍要结合合适的 token attention、外部检索或状态缓存。它与 KDA 的递归状态、DeepSeek V4 的 mHC 解决的是不同问题：AttnRes 沿深度选择层表示，KDA 沿序列维护递归状态，mHC 约束残差映射矩阵的几何性质。

## 面试追问

**问：AttnRes 和普通 self-attention 有什么区别？**  普通 self-attention 通常沿序列位置建立 token 两两关系；AttnRes 沿网络深度访问历史层表示，候选集合是层输出，不是 token 位置。

**问：为什么要用伪查询，而不是直接用当前 hidden state？**  论文的伪查询与前向计算解耦，使同一块中的查询可以批量计算，便于减少内存 I/O；代价是查询不是当前 token 内容的直接函数。

**问：Block AttnRes 为什么能降低通信？**  它只在块边界保留块级表示，跨 pipeline stage 传输的历史条目从层级数量降到块级数量；块内仍需顺序更新部分和。

**问：AttnRes 是否一定比标准残差好？**  不能这样绝对化。它增加了表达和系统复杂度，效果依赖深度、块数、训练规模、初始化、硬件和任务；应通过同预算消融实验比较。

## 小练习

1. 手算三个二维 value 在两个不同 query 下的 softmax 权重，并验证权重和为 1。
2. 将代码扩展为 Block AttnRes：块内维护 partial sum，块间保存 completed blocks。
3. 设计一个消融：固定参数量和训练 token，只改变块数 `N`，比较 loss、梯度范数、吞吐和显存。
4. 画出 Full AttnRes、Block AttnRes 和标准残差的跨 stage 通信量，说明 `L`、`N`、pipeline stage 数量之间的关系。

## Kimi K3 配置补证（2026-09-18）

Kimi K3 技术报告已明确给出 8 个 Block AttnRes block、每 block 12 层，并把 embedding representation 作为额外的 block-level representation。这个配置属于 K3 report，不是 Attention Residuals 论文所有实验的固定配置；本章前文的论文实验数字也不是 K3 实测结果。
