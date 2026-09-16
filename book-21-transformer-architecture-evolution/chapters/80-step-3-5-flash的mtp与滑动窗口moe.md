# Step 3.5 Flash：MTP-3、滑动窗口和 11B 激活参数的协同设计

> 资料来源：[StepFun 官方 Step 3.5 Flash 模型卡](https://huggingface.co/stepfun-ai/Step-3.5-Flash) 与链接的技术报告 `arXiv:2602.10604`，核验日期：2026-09-09。吞吐和 benchmark 均属于发布方协议下的结果。

## 为什么“快”是 Agent 的能力

普通聊天经常只生成一两段文字；代码 Agent 却要反复读取文件、执行测试、解析错误，再生成下一步动作。如果每个 token 都要等待一次完整前向，长任务的等待时间会迅速累积。因此 StepFun 把模型系统设计成三个互相配合的部分：稀疏 MoE 减少每 token 的主干计算，MTP-3 一次提出多个后续 token，3:1 sliding-window/full attention 混合降低长上下文成本。

## 模型卡给出的规格

Step 3.5 Flash 标为 Apache 2.0 开源模型。模型卡给出约 196.81B 总参数、约 11B 每 token 激活参数、45 层 Transformer、4096 hidden size、128,896 词表和 256K context。每层有 288 个 routed experts 与 1 个 shared expert，token 路由选择 top-8 experts。

“196B 模型、11B 执行”是 MoE 的两本账：总参数决定权重和分片，激活参数决定一次 token 的主要矩阵乘规模。专家并行仍会产生 token dispatch、all-to-all 和负载不均衡，不能把 11B 当作单卡内存需求。

## 3:1 Sliding Window Attention

模型卡称每三个 sliding-window attention 层配一个 full-attention 层。局部层只看窗口内 token，global 层周期性地重新连接远处信息。若窗口大小是 `w`、序列长度是 `n`，局部层的可见 pair 近似 `O(nw)`，full 层仍接近 `O(n^2)`；混合比例降低平均成本，却不等于所有层都变成线性复杂度。

可以用一个抽象可见性函数表示：

```math
V_l(i,j)=
\begin{cases}
1,&l\bmod 4=0\\
1,&l\bmod 4\ne0\;\land\;0\le i-j<w\\
0,&\text{otherwise}.
\end{cases}
```

这里 `l mod 4=0` 只是表达“四层中有一层 full”的教学约定，真实层编号、窗口和实现细节需以模型配置为准。局部层节省计算的同时，会让信息跨层传播；global 层的位置和数量会影响远程检索质量。

```python
def visible_pairs(seq_len, window, full_every=4):
    local = 0
    full = 0
    for layer in range(full_every):
        for i in range(seq_len):
            if layer % full_every == 0:
                full += i + 1  # causal full attention
            else:
                local += min(i + 1, window)
    return {"full_pairs": full, "local_pairs": local}


print(visible_pairs(seq_len=16, window=4))
```

示例只统计 causal 可见 pair，没有实现 softmax、KV cache 或 GPU kernel。评估时应分别测窗口内复制、跨窗口检索和长代码变量追踪，避免只用平均 perplexity 判断局部注意力是否足够。

## MTP-3 为什么能降低等待

自回归模型通常一次确定一个 token。MTP 头让模型在一次主干前向后提出多个候选 token，再由验证逻辑确认可接受的前缀。官方卡称其使用 3-way Multi-Token Prediction，并报告典型 100–300 tok/s、单流编码峰值约 350 tok/s；这些数字依赖硬件、批大小、量化和后端。

推测解码的核心指标不是“预测几个”，而是平均接受长度：

```math
\text{effective tokens per target call}=1+\mathbb{E}[A],
```

其中 `A` 是被目标模型接受的连续草稿长度。草稿不准时会频繁回退，额外的验证和状态清理可能抵消收益。模型卡还特别注明当前 vLLM 页面中的完整 MTP3 支持仍在集成，因此模型具备 MTP 头与某个后端已经实现全部加速路径是两件事。

## Agent 评测必须记录 Context Manager

Step 3.5 Flash 模型卡给出 BrowseComp、SWE-bench Verified 和 Terminal-Bench 2.0 等结果，并说明 BrowseComp 的 Context Manager 在有效上下文超过阈值时会重置并重新开始 agent loop。这一策略是 harness 的状态管理，不是模型本身的记忆机制。

一个长任务记录至少要包含：模型 revision、reasoning 参数、工具 schema、上下文压缩阈值、重启次数、硬件、后端版本和最终 artifact。没有这些字段，无法判断失败来自模型推理、上下文溢出、工具解析还是重启策略。

## 与 MoE 路由的系统耦合

Step 3.5 的 288 routed experts、top-8 路由和 shared expert 会影响 expert parallel 的通信；MTP 又会改变每轮前向中 token 数和验证节奏；sliding-window 则改变 KV cache 的布局。三者不能分别只看论文中的单模块指标。

例如，若 top-8 路由造成某些专家过载，MTP 提高的 token 候选数可能放大 dispatch 峰值；若窗口层的 KV cache 采用环形缓冲区，global 层又需要更长的历史布局，serving engine 需要维护两种 cache 生命周期。实际部署应测量 all-to-all、cache hit、接受率、p95 和每个请求的有效 token 成本。

## 公开性能数字的正确读法

模型卡列出 SWE-bench Verified 74.4%、Terminal-Bench 2.0 51.0% 等结果，并与其他模型比较。阅读时要固定：测试集版本、是否使用 Context Manager、工具权限、超时、重试、上下文重启、采样参数和硬件。发布方表格中的星号和 reproduced 分数也说明不同来源的协议并不总是一致，不能将所有数字拼成一张绝对排行榜。

## 面试追问

**问：3:1 SWA 是否把复杂度降成 O(n)？** 只有局部层的 pair 数近似 `O(nw)`；周期性的 full 层仍接近 `O(n²)`，整体是混合成本。

**问：MTP-3 一定带来三倍吞吐吗？** 不一定。收益由草稿接受率、验证成本、后端支持、批处理和同步开销共同决定。

**问：11B active parameters 是否意味着 11B 显存？** 不是。权重总量、专家分片、KV cache、通信 buffer 和运行时 workspace 仍需纳入显存账本。

**问：Context Manager 属于模型架构吗？** 它属于 Agent harness 的上下文管理策略；模型卡把该策略用于评测，不能据此宣称模型拥有永久记忆。

## 小练习

1. 修改代码统计不同窗口大小和 full 层比例下的 causal pair 数，画出相对 full attention 的比例。
2. 构造接受率分别为 0.2、0.5、0.8 的 MTP toy 模拟，比较目标模型调用次数。
3. 为 MoE top-8 路由加入容量因子，记录 overflow、负载方差和 all-to-all token 数。
4. 设计一个带 Context Manager 重启的长任务评测，分别报告模型失败、工具失败、上下文重启和超时。
