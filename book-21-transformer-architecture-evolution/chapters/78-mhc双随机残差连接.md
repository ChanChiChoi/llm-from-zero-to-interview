# mHC：用双随机矩阵约束残差流

> 资料来源：DeepSeek V4 技术报告 `arXiv:2606.19348` 源码（核验日期 2026-09-09）。mHC 是 DeepSeek V4 报告中的 Manifold-Constrained Hyper-Connections。它和 Attention Residuals 都关注深层信号传播，但机制不同：mHC 约束残差映射的几何结构，AttnRes 沿深度选择历史表示。

## 残差连接为什么需要稳定性分析

标准残差写成：

```math
x_{l+1}=x_l+F_l(x_l).
```

它让信息和梯度拥有一条直接路径，但在更复杂的 Hyper-Connections 中，残差流可能不再是一个向量简单加上一个分支。多个 residual stream 可以通过矩阵混合，层与层之间既要保留表达能力，又要避免信号范数在深度方向失控。

可以把残差流想成几条并行水管。每一层都能把水从一条管道分配到其他管道。如果分配矩阵任意放大某些方向，经过很多层后，水压可能爆炸；如果不断压低某些方向，信号又会逐渐消失。mHC 的思路是限制这张“分配表”的几何性质。

## Hyper-Connections 的抽象形式

DeepSeek V4 报告将一层写成：

```math
X_{l+1}=B_lX_l+C_l\mathcal{F}_l(A_lX_l),
```

其中 `X_l` 表示多路残差流，`A_l` 把输入混合后送入主变换 `F_l`，`C_l` 把变换结果写回残差流，`B_l` 则直接混合上一层的残差。与普通残差相比，这种形式允许更丰富的跨流交互，但也引入了传播稳定性问题。

mHC 的关键约束作用在 `B_l`：它把 `B_l` 投影到双随机矩阵集合，也就是 Birkhoff polytope：

```math
\mathcal{M}=\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}=\mathbf{1},\;\mathbf{1}^\top M=\mathbf{1}^\top,\;M\ge 0\}.
```

“行和为 1、列和为 1、元素非负”意味着每个输出流接收的总混合权重受控，每个输入流分配出去的总权重也受控。它不是简单要求矩阵元素小，而是对整体映射结构加约束。

## 为什么双随机矩阵有帮助

若 `B` 和 `C` 都是双随机矩阵，那么 `BC` 仍然是双随机矩阵。深层网络连续使用这类映射时，传播不会因为矩阵连乘而任意破坏行列总量。这个封闭性提供了一个稳定性直觉：残差流可以重新组合，但不会凭空制造或删除总混合质量。

这不是说双随机矩阵保证每个神经元的范数永远不变。它约束的是流之间的线性混合结构；非线性层、写入矩阵 `C_l`、归一化和参数更新仍会影响实际激活和梯度。正式分析必须区分“结构约束带来的稳定性倾向”和“训练全过程的绝对稳定保证”。

## 一个二维例子

下面两个矩阵都满足行列和为 1：

```math
B_1=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
B_2=\begin{bmatrix}0.7&0.3\\0.3&0.7\end{bmatrix}.
```

`B_1` 保持两条流不变，`B_2` 则把每条流的 70% 留在原通道、30% 混到另一通道。无论连续乘多少次，矩阵仍处在双随机集合中。相比之下，任意矩阵 `[[1.4, 0.2], [0.1, 1.3]]` 会同时改变行列总量，连续堆叠时可能放大某些方向。

## Sinkhorn 型投影的直觉

训练得到的原始映射 `\tilde B_l` 通常不会天然双随机。报告源码描述通过交替行归一化和列归一化，把它近似投影到双随机集合：

```text
M^(0) = exp(raw_logits)  # 元素为正，实际实现还需要稳定化
M^(t) = normalize_columns(normalize_rows(M^(t-1)))
```

每次行归一化让行和接近 1，列归一化又会改变行和，因此要交替多轮。得到的矩阵通常只是在有限迭代和数值精度下接近双随机；正式实现需要说明迭代次数、稳定化和梯度传播。

## 零依赖教学实现

```python

def normalize_rows(matrix):
    out = []
    for row in matrix:
        total = sum(row) or 1.0
        out.append([x / total for x in row])
    return out


def normalize_columns(matrix):
    rows, cols = len(matrix), len(matrix[0])
    sums = [sum(matrix[i][j] for i in range(rows)) or 1.0
            for j in range(cols)]
    return [[matrix[i][j] / sums[j] for j in range(cols)]
            for i in range(rows)]


def sinkhorn(raw, steps=20):
    matrix = [row[:] for row in raw]
    for _ in range(steps):
        matrix = normalize_rows(matrix)
        matrix = normalize_columns(matrix)
    return matrix


projected = sinkhorn([[1.4, 0.2], [0.1, 1.3]])
print("row sums:", [round(sum(row), 4) for row in projected])
print("column sums:", [round(sum(projected[i][j]
                               for i in range(2)), 4)
                       for j in range(2)])
```

示例传入的是严格正的矩阵，不是任意实数 logits。若直接对负数矩阵归一化，非负约束不会自动成立；零行或零列也不能靠把分母改成 1 修复。生产实现从参数化分数构造正矩阵，再做有限步归一化。

这个示例用于说明投影过程，不是报告中的完整 mHC kernel。实际 mHC 还要计算 `A_l`、`C_l`、动态参数、残差流布局和 Transformer 主分支，并考虑训练通信和 fused kernel。

## mHC 和其他残差改进的区别

- **标准残差**：直接相加，结构简单，几乎没有额外混合参数。
- **Hyper-Connections**：允许多路 residual stream 通过矩阵交互，表达力更强，但稳定性和系统开销更复杂。
- **mHC**：在 Hyper-Connections 的残差映射上加入双随机流形约束，目标是稳定深层传播。
- **Attention Residuals**：对不同深度的历史表示做输入相关加权，解决的是“应该读取哪些层”的选择问题。
- **KDA**：沿序列维护有限状态，解决的是长上下文 token 历史的压缩和更新问题。

同一个模型可以组合这些思想，但它们作用的对象不同，不能把“都改了 residual”当成相同技术。

## 工程代价和边界

mHC 增加了矩阵混合、投影、通信和激活保存需求。DeepSeek V4 报告源码提到通过 fused kernels、重计算和调整 pipeline overlap 控制开销，并报告其特定实现中 mHC 增加的 wall-time 约为 pipeline stage 的 6.7%。这个数字依赖模型、并行策略和硬件；教学时应把它作为案例，而不是通用常数。

在训练系统中，矩阵约束还会影响 ZeRO 分片、激活检查点和跨 stage 通信；在推理系统中，多路残差流会影响 kernel 融合和内存布局。判断 mHC 是否值得采用，需要同时比较 loss、梯度稳定性、吞吐、显存和通信，而不能只看单个 benchmark。

## 面试追问

**问：双随机约束为什么比限制矩阵范数更具体？**  范数约束只控制整体大小，双随机约束还控制每个输入/输出流的混合总量和非负性，提供更明确的流量守恒结构。

**问：Sinkhorn 迭代后一定严格双随机吗？**  只在理想收敛和数值精度条件下接近；实际是有限步近似，需要检查行列和误差。

**问：mHC 会不会阻止表达能力？**  它只约束直接残差混合矩阵，主变换 `F_l`、输入混合和写回矩阵仍提供表达能力；但约束可能带来优化和实现成本。

**问：mHC 与 AttnRes 能否同时使用？**  理论上作用对象不同，可以组合，但需要重新评估残差流、深度聚合、初始化和通信开销，不能直接叠加论文结论。

## 小练习

1. 实现 `3×3` raw matrix 的 Sinkhorn 投影，记录每轮行列和误差。
2. 比较双随机矩阵和任意矩阵连续相乘后的谱范数与行列和。
3. 为一个两路 residual stream 写出 `A/B/C/F` 的张量形状。
4. 设计 mHC 与标准残差的同预算消融，记录 loss、梯度范数、激活峰值和 pipeline 通信量。
