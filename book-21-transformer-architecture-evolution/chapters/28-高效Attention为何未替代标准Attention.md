# 第二十八章：高效 Attention 为什么没有完全替代标准 Attention

## 28.1 先修正一个错误问题

“哪个高效 attention 会彻底替代标准 attention”往往不是最好的问题。更准确的问题是：在什么任务、长度、硬件、训练规模和质量门槛下，减少显式 token-token 交互的方案值得采用；哪些工作由局部/线性/状态路径完成，哪些工作必须保留精确 global attention。

标准 attention 的优势不仅是公式简单。它有成熟的 dense GEMM kernel、稳定的训练生态、强的内容寻址能力、可解释的 token 级路径和大量预训练 checkpoint。替代路线要同时赢过质量、训练、推理、硬件和开发风险，难度很高。

## 28.2 理论复杂度不是最终答案

把标准 attention 写成 `O(T^2)`、替代方案写成 `O(T)`，只比较了序列长度的渐近项。真实时间还包括：

~~~math
T_{\mathrm{wall}}=
T_{\mathrm{matmul}}+
T_{\mathrm{memory}}+
T_{\mathrm{communication}}+
T_{\mathrm{launch}}+
T_{\mathrm{scheduler}}
~~~

标准 attention 的大矩阵乘法可能接近 GPU 峰值；线性/递归方案的小矩阵更新可能低利用率；稀疏方案要付索引和 gather；混合方案要付路径切换和状态管理。序列短或 batch 小时，常数项可能主导。

## 28.3 能力损失集中在哪些任务

高效 attention 通常最容易保留局部语言建模和主题级任务，最容易暴露问题的是：

1. 远处精确复制。
2. 多个相似实体的绑定。
3. 冲突版本的条件选择。
4. 根据后来的 query 重新解释早期证据。
5. 任意位置引用和工具参数恢复。

这些任务依赖动态、内容条件的历史寻址。固定状态或静态稀疏图可能有足够的平均能力，却缺少某一条关键证据路径。

## 28.4 训练生态的惯性

大量预训练模型、数据管道、并行策略、量化库、推理引擎和评测工具围绕标准 Transformer 建立。替代架构不仅要重新训练 backbone，还要验证 tokenizer、position、checkpoint、混合精度、梯度检查点、分布式通信和 serving。

如果一个新架构只在小模型或短任务上优于 dense baseline，迁移到大规模训练时可能遇到 kernel、优化器和数据分布问题。生态不是保守借口，而是总成本的一部分。

## 28.5 长上下文为什么常采用混合路径

局部/递归路径提供稳定的线性或近线性扫描，全局 attention 提供精确跳转。设 `L` 层中有 `L_g` 层 global，其他为 local/state，可用一个粗略预算表示：

~~~math
C_{\mathrm{total}}\approx L_g C_{\mathrm{global}}+
(L-L_g)C_{\mathrm{cheap}}
~~~

只要少量 global 层能承担高价值跨段任务，就可能得到质量/成本折中。代价是两种状态、不同 mask、cache、训练和 serving 生命周期都要维护。

## 28.6 一个决策矩阵

| 需求 | 更偏标准/global | 更偏高效/local/state |
|---|---|---|
| 任意原文引用 | 是 | 需外部检索或显式出口 |
| 纯流式长序列 | 成本高 | 更有吸引力 |
| 大 batch prefill | dense kernel 强 | 需验证 kernel |
| 多实体冲突推理 | 更稳 | 需长程专项评测 |
| 边缘设备内存 | KV 压力大 | 固定状态/窗口有利 |
| 生态和 checkpoint | 成熟 | 迁移成本高 |

这是起始假设，不是结论。最终选择要用相同质量验收条件和端到端 workload 验证。

## 28.7 质量、成本和风险的验收条件

可以定义一个架构上线条件：

~~~math
G=
\mathbb{1}[Q_{\mathrm{task}}\ge Q_0]\,
\mathbb{1}[R_{\mathrm{recall}}\ge R_0]\,
\mathbb{1}[P99\le SLO]\,
\mathbb{1}[C_{\mathrm{success}}\le C_0]
~~~

其中 `R_recall` 不是普通检索指标，而是关键证据被正确利用的比例。任何一项失败都不能用平均 tokens/s 掩盖。若新架构质量只差一点，也要问那一点是否落在高风险任务上。

## 28.8 一个小型 Pareto 实验

~~~python
def gate(quality, evidence_recall, p99_ms, cost,
         q0=0.90, r0=0.92, p99_0=120, cost0=1.0):
    return {
        "quality_ok": quality >= q0,
        "evidence_ok": evidence_recall >= r0,
        "latency_ok": p99_ms <= p99_0,
        "cost_ok": cost <= cost0,
        "ready": (quality >= q0 and evidence_recall >= r0
                  and p99_ms <= p99_0 and cost <= cost0),
    }


print(gate(0.91, 0.95, 90, 0.82))
print(gate(0.93, 0.88, 60, 0.60))
~~~

第二个方案更快更便宜，但证据 recall 不通过验收，不能直接上线需要精确引用的业务。这个例子提醒我们，架构比较必须让业务关键指标拥有独立门槛。

## 28.9 为什么标准 attention 仍然有优化空间

标准 attention 的瓶颈并不只有算法形式。GQA/MQA 减少 KV head，FlashAttention 减少中间访存，PagedAttention 减少 cache 碎片，prefix cache 复用重复前缀，量化降低带宽，continuous batching 提高设备利用率。很多业务不需要替换 attention，只需要把 serving 层做对。

这也是“完全替代”难以发生的原因：标准 attention 的缺点正在被多层工程优化拆解。替代架构必须在仍未解决的场景上提供足够大的增益，才能抵消迁移成本。

## 28.10 什么时候高效路线会赢

当序列极长、请求持续流入、局部性强、精确随机访问需求低、硬件支持对应 kernel、状态可可靠迁移且质量检查通过时，SSM、linear attention、sliding window 或混合路线可能赢得明显优势。音频、传感器、日志、基因组和部分视频任务可能比通用文档推理更适合。

“赢”也可能表现为只承担一部分层或一类请求，而不是替代整套模型。架构路线的价值可以体现在更低功耗、更稳定流式状态或更高并发，而不是榜单上的单一准确率。

## 28.11 机制与边界：用任务条件互信息理解保留什么

固定状态或稀疏连接保留的是输入的某种摘要。若未来任务查询 `q` 对历史 `H` 的依赖只通过低维统计量表达，那么压缩是合理的；若任务需要从 `H` 中随机选择细节，固定摘要很可能不够。可以把目标写成让状态 `S` 最大化：

~~~math
I(S;Y\mid Q)
~~~

同时控制状态成本 `C(S)`。这是分析框架而非可直接训练的完整目标。它说明架构选择应由未来查询分布决定：不是“历史本身要保存多少”，而是“未来会用到什么”。

## 28.12 面试追问、误区与练习

**问：为什么 Mamba/linear attention 没有立刻完全取代 Transformer？**

标准回答：它们在长流成本和状态大小上有优势，但精确内容寻址、ICL、长程组合、训练/推理 kernel、生态和 serving 状态管理仍有 trade-off。实际更可能先出现 hybrid，而不是一次性替代。

**问：怎样证明一个高效 attention 值得上线？**

标准回答：在相同模型/数据/硬件和真实 workload 下，分别测局部、远距、精确引用、多证据、TTFT、TPOT、p99、峰值显存、单位成功成本和恢复一致性，并设置质量与系统双验收条件。

常见误区包括只比较 Big-O、把论文 speedup 当生产结果、只测 perplexity、忽略失败任务的业务价值，以及把 hybrid 说成纯替代。

练习：为一个 1M 日志分析服务选择 dense、sliding window、SSM 或 hybrid，写出证据任务、系统指标、迁移成本和回退策略。

## 28.13 生态和验证成本

标准 attention 拥有成熟的 kernel、量化、并行、监控和调试工具。替代路线即使论文结果漂亮，也要重新建立 checkpoint、梯度、batch、cache、故障恢复和评估基础设施。

当一个系统的主要成本来自研发和回归，而不是 attention FLOPs 时，成熟标准路径可能仍然是更好的工程选择。真正的替代需要在目标工作负载上覆盖质量、吞吐、p99、故障和维护成本。

## 28.14 何时应该采用高效路线

如果输入是连续流、任务允许压缩、上下文远大于 GPU 显存，或者显式 attention 成为明确的主要瓶颈，高效路线更有价值。反之，如果任务要求随机精确引用、数据量中等且成熟 kernel 已经足够快，复杂替代可能得不偿失。

可以把决策写成验收条件：任务质量不下降，单位成功成本下降，回归和恢复通过，运行时有可观测 fallback。缺少任一项，都不应仅凭 Big-O 迁移。

## 28.15 近似、稀疏、状态和 IO 优化不是同一层

高效 attention 至少有四个层次。近似方法改变 attention 的数学表达；稀疏方法减少可见边；状态方法把历史压缩成递归统计；FlashAttention 这类方法保持数学结果，优化 tile、访存和中间矩阵。它们的质量和系统风险不能用同一指标替代。

例如 FlashAttention 可能在相同语义下改善显存和速度，线性 attention 可能降低历史存储却改变精确检索能力；滑窗可能通过任务结构节省边，近似 kernel 可能引入 seed 方差。比较前先明确优化发生在哪一层。

## 28.16 迁移决策的成本模型

采用新 attention 路线的总收益不应只写成单次推理速度。可以用以下框架估算：

~~~math
\Delta V
=V_{\mathrm{quality}}
+V_{\mathrm{latency}}
+V_{\mathrm{capacity}}
-C_{\mathrm{migration}}
-C_{\mathrm{regression}}
-C_{\mathrm{fallback}}
~~~

迁移成本包括重新训练、kernel、量化、监控、调度、评估和开发者工具；回归成本包括精确引用、长任务、工具调用和多模态任务的质量风险。只要高价值任务仍需大量回退，理论收益就可能被抵消。

## 28.17 什么时候标准 attention 仍是理性选择

当上下文长度在成熟 paged/FlashAttention 系统的可承受范围内，任务要求任意精确引用，且团队更看重模型生态和可调试性时，标准 attention 可能是更稳妥的基线。它不是没有成本，而是成本已经有成熟的 allocator、量化和 profiling 工具支持。

高效路线更适合明确的长流、低内存、局部性或允许近似的任务。无论选择哪条路线，都应保留 dense baseline、质量回退和长任务回放，避免把架构迁移变成不可逆赌注。

## 28.18 迁移成本本身就是模型成本

一种高效 attention 即使在理想 kernel benchmark 上更快，也可能需要新编译器、新 cache layout、新量化路径、新 checkpoint 转换和新的 debug 工具。迁移成本包括工程人力、回归集、灰度时间、故障回滚和生态兼容；如果部署规模小，理论节省可能无法覆盖这些固定成本。

可以把采用决策写成：

```math
\mathrm{Adopt}
\iff \Delta C_{\mathrm{run}}
>C_{\mathrm{migration}}+C_{\mathrm{risk}}+C_{\mathrm{maintenance}}
```

这里的 `C_risk` 不只是金钱，还包括质量下降、协议错误和故障恢复成本。对高风险应用，即使运行成本略高，成熟 dense path 也可能是理性选择。

## 28.19 一个分阶段采用流程

第一阶段用 reference 实现验证数学和任务质量；第二阶段用目标 runtime 测 kernel、显存和变长 batch；第三阶段加入 cache、preemption、量化和多租户；第四阶段做灰度、回滚和故障注入。每阶段都要有停止条件，不能因为论文复杂度漂亮就直接跳到生产。

高效路径应明确 native、adapted 和 fallback 状态。若某个模型只有部分层支持新结构，服务端要知道哪些请求会走混合路径；若 grammar、speculative 或多模态输入不支持，应显式拒绝或降级，不能返回一个看似成功但语义变化的结果。

## 28.20 标准 attention 仍然可能是最优解

当任务依赖任意远程证据、上下文中等、成熟 kernel 已能满足 SLO、质量回退代价高或生态需要兼容多个模型时，标准 attention 仍可能占据 Pareto 前沿。它也持续通过 FlashAttention、GQA/MLA、KV quantization、prefix cache 和更好的调度降低成本。

真正的选择不是“新架构对旧架构”，而是目标任务和资源下的完整系统比较。只要新方法不能在相同质量验收条件、SLO、协议和维护成本下形成优势，就不应为了追逐复杂度而替换成熟路径。

## 28.21 把“替代”拆成四个问题

“高效 attention 为什么没有替代标准 attention”并不是一个单一的技术问题。至少要拆成：是否能替代训练计算图，是否能替代离线长上下文质量，是否能替代线上 serving runtime，以及是否能替代现有模型生态和工具链。一个方法可能在第一项不占优，却在第三项作为 kernel 优化非常成功；也可能在长流任务上很好，却不适合需要任意引用的通用助手。

对应地，实验要有四套门：matched model quality、hardware kernel、end-to-end serving 和 migration governance。前两项通过不代表后两项通过。尤其不能把某个论文模型的训练结果、另一个 runtime 的吞吐和第三个产品的上下文能力拼成一条因果链。

最终选择可以写成一个带约束的优化问题：在质量、p99、状态恢复和安全门通过的候选中，比较运行成本与迁移成本。若高效路线必须频繁回退到 dense path，或者每次模型升级都需要重写 tokenizer、cache 和工具协议，它的“替代”就可能只发生在局部层，而不是整套系统。

## 28.22 迁移前必须做的反事实

在替换标准 attention 前，至少保留三组对照：原始 dense attention、只换 kernel/serving 的优化版本、候选高效 attention。三者使用同一模型规模、数据、长度分布和质量验收条件，再分别测 prefill、decode、长上下文证据、结构化输出、恢复和成本。

如果只换 FlashAttention 就已消除大部分瓶颈，换架构的额外收益必须抵消状态协议、checkpoint、kernel、调度和生态迁移成本；如果高效路径只在超长输入上赢，应考虑按请求长度路由，而不是让所有短请求承担近似误差。

## 28.23 高效 Attention 的收益来源

Longformer、BigBird、Performer、Mamba 和 FlashAttention 的论文分别讨论了不同层次的优化。它们的实验结论不能跨模型、硬件和任务直接拼接；任何“替代”判断都应回到可复现的消融和端到端压测。

高效 attention 没有完全替代标准 attention，并不表示这些路线失败。它们把“全历史显式读取”分解成局部、状态、近似和全局路径，为混合架构和特定 workload 提供了更好的选择。

## 28.24 先分清优化发生在哪一层

“高效 attention”常被当成一个总称，但至少包含四种不同变化。第一种是数学近似，例如 kernel feature map 或 LSH，改变了相似度计算本身；第二种是连接稀疏，只保留局部、全局或块级边；第三种是状态压缩，用固定 state 替代显式历史；第四种是 IO 和 kernel 优化，在保持 dense 语义的同时减少中间张量和无效显存访问。

四者的风险不同。数学近似可能改变输出分布，稀疏连接可能使证据不可达，状态压缩可能损失可寻址历史，IO 优化主要承担数值一致性和硬件兼容风险。把 FlashAttention 的收益和 SSM 的收益放在同一列“复杂度降低”中，会让读者误以为它们可以用同一个质量指标判断。

一个更清楚的分解是：

```math
C_{\mathrm{system}}
=C_{\mathrm{math}}+C_{\mathrm{connect}}
 +C_{\mathrm{state}}+C_{\mathrm{io}}+C_{\mathrm{protocol}}.
```

这个式子不是严格的性能计数，而是架构评审的账本。新方案至少要说明减少了哪一项、增加了哪一项，以及新增成本是否会在 batch、长上下文、量化、回滚和多租户下放大。

## 28.25 用 roofline 直觉解释“理论更快却没变快”

计算复杂度下降后，系统可能从 compute-bound 变成 memory-bound 或 launch-bound。若每个 token 的状态更新很小，但需要频繁读写大 state，GPU 算力没有被充分利用；若稀疏 pattern 不规则，索引和 kernel launch 可能成为主耗时；若 batch 很小，理论 FLOPs 节省更难摊薄固定开销。

可以用教学化的 roofline 上界表达：

```math
T\ge\max\left(\frac{F}{P_{\mathrm{compute}}},
\frac{M}{B_{\mathrm{memory}}},
N_{\mathrm{kernel}}T_{\mathrm{launch}}\right).
```

高效 attention 可能显著减少 `F`，却增加 `M` 或 `N_kernel`。因此 benchmark 应报告 FLOPs 只是起点，还要测显存读写、kernel 数、occupancy、batch 利用率和实际 p99。这个解释比“GPU 没有优化好”更具体，也能指导是换 kernel、改 block，还是换架构。

## 28.26 质量回退必须按错误类型拆分

平均 perplexity 或总体准确率不能说明高效路径是否适合一个真实系统。对长上下文助手，至少要区分：局部语法、远距数字复制、冲突实体、多个证据合并、引用支持、结构化输出、工具参数和安全拒答。不同优化对这些任务的影响方向可能相反。

例如状态压缩可能保留主题摘要，却丢掉精确版本号；稀疏 mask 可能提高局部代码速度，却漏掉跨文件定义；近似 kernel 可能平均误差很小，却在尖锐 attention 分布上改变 top-1 证据。每类错误应绑定可观察的路径指标：证据是否可达、是否被状态写入、是否在输出中被读取、kernel 是否 fallback。

发布分数可以采用硬性条件而非简单加权平均：

```math
G=G_{\mathrm{quality}}G_{\mathrm{protocol}}G_{\mathrm{safety}}
 G_{\mathrm{latency}}G_{\mathrm{recovery}}.
```

只要关键项为零，整体就不放行。这样可以避免“平均吞吐提升”抵消一个不可接受的工具副作用或引用错误。

## 28.27 局部替代比全面替代更现实

一个成熟系统可以按长度、任务和风险做路由：短输入走标准 dense kernel，长流走 state/SSM，需要精确引用的请求走 dense 或 RAG，低风险批处理允许近似，结构化工具请求使用更严格的验证和回退。这种组合不要求某一种架构赢得全部 workload。

路由本身也有成本。它需要预测请求类型、预估长度、保留多种 cache/schema、维护不同监控和回滚路径；如果路由错误，用户可能在不同请求上看到不一致的质量。路由策略应在 offline replay、shadow、canary 和故障演练中验证，并记录每次选择的原因。

所以“没有替代标准 attention”更准确的说法是：没有一种新方法在所有任务、硬件、协议和治理条件下同时支配 dense baseline。但在局部 workload 上，近似、稀疏、状态和 IO 优化都可能形成优势；书写和面试时要报告这个条件，而不是用绝对结论替代实验。
