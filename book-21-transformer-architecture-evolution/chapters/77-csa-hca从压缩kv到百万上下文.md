# CSA/HCA：从压缩 KV 到百万上下文

> 资料来源：DeepSeek V4 官方模型卡、V4 Preview 公告与技术报告 `arXiv:2606.19348`（核验日期 2026-09-09）。本章解释报告公开的架构，不把发布方的 FLOPs 比例外推为所有硬件的保证。

## 先算一笔 KV Cache 账

全注意力在生成时需要保留历史 key/value。假设有 `T` 个 token、每个 KV 条目有 `d` 个元素、使用 `b` 字节存储，那么单层缓存近似为 `2Tdb`。当 `T` 从 8K 变成 1M 时，缓存不是“多一点”，而是按序列长度线性增长。多层、多请求和 batch 会进一步放大显存压力。

单纯把上下文窗口字段改成 1M，只说明接口允许更长输入；真正困难是让模型在这个长度下仍有可接受的检索质量、延迟、显存和成本。DeepSeek V4 的 CSA/HCA 试图在 KV 表示层面压缩历史，再在压缩表示上做选择或注意力。

## CSA：先压缩，再稀疏选择

Compressed Sparse Attention（CSA）先把连续 token 的 KV 沿序列维压缩。报告源码描述为：每 `m` 个 token 汇聚成一个压缩条目，序列长度大致变成原来的 `1/m`。随后，DeepSeek Sparse Attention 的 indexer 为每个 query 计算压缩条目分数，只保留 top-k 压缩块进入核心注意力。

这包含两个不同动作：

1. **压缩**减少候选数量，但可能损失块内细节。
2. **稀疏选择**进一步减少每个 query 实际访问的候选。

如果把两步混成“稀疏注意力”，就无法分析错误来源：压缩可能让相关 token 在汇聚时被稀释，top-k 可能在选择阶段漏掉相关块。

一个教学化的压缩可以写成：

```math
c_j=\sum_{r=0}^{m-1}z_{j,r}x_{jm+r},
```

其中 `x` 是原始 KV 条目，`z` 是压缩权重。真实报告还定义了压缩 KV、indexer key、因果可见性和局部滑动窗口；上式只是帮助理解“多个 token 变成一个条目”的简化表达。

## HCA：更激进压缩，但保留 dense attention

Heavily Compressed Attention（HCA）使用比 CSA 更大的压缩跨度 `m'`，把更多 token 汇聚成单个条目，但在压缩后的条目上保持 dense attention。它没有 CSA 的 top-k 稀疏选择，因此每个 query 会访问可见的压缩条目集合。

CSA 更像“较温和压缩 + 稀疏检索”，HCA 更像“重压缩 + 密集读取”。两者可以交错放在不同层，用不同的误差和计算路径换取整体效率。报告还保留滑动窗口分支，为最近 token 提供未被大幅压缩的局部信息。

## 为什么需要局部滑动窗口

压缩条目适合远距离趋势和粗粒度检索，却可能损失刚刚出现的变量、标点、局部语法或代码缩进。滑动窗口让 query 同时访问最近的一段原始 KV，形成“远处压缩、近处精细”的组合。因果约束要求当前位置只能看到此前已经完成压缩的块，以及允许的局部历史，不能因压缩实现而偷看未来 token。

## KV Cache 不再是单一数组

在普通 GQA 中，不同层通常有相对统一的 KV 布局；CSA/HCA 混合后，不同层的压缩倍率、indexer 维度、滑动窗口长度和未完成尾部都可能不同。DeepSeek V4 报告因此描述了异构 KV Cache：

- CSA/HCA 的压缩 KV。
- 稀疏选择所需的 indexer 状态。
- 滑动窗口最近 token。
- 尚未达到压缩块边界的尾部 token。
- 共享前缀和磁盘复用所需的状态。

Serving engine 不能只按“每 token 固定字节数”分配缓存，而需要记录不同层、不同请求和不同缓存段的形状与淘汰规则。

## 一个最小压缩与 top-k 示例

```python

def compress(values, weights, block_size):
    compressed = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        w = weights[start:start + block_size]
        z = sum(w) or 1.0
        compressed.append(sum(x * a for x, a in zip(block, w)) / z)
    return compressed


def top_k_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

values = [1.0, 2.0, 10.0, 11.0, 3.0, 4.0]
compressed = compress(values, [1.0] * len(values), block_size=2)
scores = [0.2, 1.4, 0.7]
chosen = top_k_indices(scores, k=2)
print("compressed:", compressed)
print("chosen blocks:", chosen)
```

这个示例没有实现真实的向量 KV、causal mask、indexer 或 GPU kernel，只用于演示压缩和选择是两个阶段。工程实现需要明确压缩权重、精度、块边界、top-k、局部窗口和 cache eviction。

## 成本指标必须绑定基线

DeepSeek V4 报告给出 1M 场景下相对 V3.2 的单 token FLOPs 和 KV Cache 比例，也给出相对 BF16 GQA8 基线的缓存比例。它们回答的是不同问题：一个是同系列版本对比，一个是与特定注意力配置对比。正式评估时至少要记录：模型版本、上下文长度、精度、batch、硬件、是否包含 indexer、是否包含滑动窗口和 cache 命中情况。

## 局限与面试追问

压缩会带来信息损失；稀疏 top-k 会带来漏检风险；异构 cache 增加调度和实现复杂度；FP4/FP8 还会引入量化误差。CSA/HCA 不是无条件替代全注意力，而是把计算和存储预算集中在更有价值的历史表示上。

**问：CSA 和 HCA 的核心区别？** CSA 在压缩后还做稀疏 top-k 选择；HCA 进行更强压缩，但在压缩条目上保持 dense attention。

**问：为什么还需要滑动窗口？** 为近期 token 保留精细信息，弥补压缩远程表示对局部细节的损失。

**问：1M context 是否意味着模型能准确检索任意位置？** 不是。窗口容量、有效检索、延迟、显存和评测分布是不同指标。

**问：Serving engine 最难的地方是什么？** 不是只把 KV Cache 做大，而是管理不同层的压缩倍率、稀疏索引、局部窗口、未完成尾部和共享前缀。

## 小练习

1. 将示例改为二维 key/value 向量，并实现按块平均压缩。
2. 加入 causal mask，验证当前位置不会读取未来压缩块。
3. 比较无压缩、CSA 风格和 HCA 风格的缓存元素数。
4. 设计一个检索评测，分别测压缩损失、top-k 漏检和局部窗口补偿。
