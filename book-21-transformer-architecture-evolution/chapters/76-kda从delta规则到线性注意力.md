# 第 76 章 Kimi Delta Attention：从可更新记忆到线性注意力

> 资料版本：Kimi Team，*Kimi Linear: An Expressive, Efficient Attention Architecture*，arXiv:2510.26692 v2（2025-11-01）。第 61 章负责 KDA 的递归公式和逐通道衰减；本章把问题移到模型层面：KDA 如何与全局 MLA、NoPE 和 MoE 组成 Kimi Linear，以及论文中的质量、缓存和速度证据各自支持到哪里。这里的层比和数字只描述 Kimi Linear 对应版本，不等同于 Kimi K3 或其他混合模型配置。[论文 v2](https://arxiv.org/html/2510.26692v2)

Kimi Linear 的关键设计不是单独把全注意力换成递归状态。它让大多数 token-mixing 层使用 KDA，再周期性插入全局 MLA；每个 token-mixing layer 后还接 MoE channel-mixing layer。这样的结构同时包含递归 state、按 token 保存的 latent cache 和稀疏专家计算。要评价它，必须分别看信息路径、训练对照和推理测量。

## 76.1 先把 Kimi Linear 拆成三条结构轴

### 76.1.1 48B 总参数和 3B 激活参数描述不同的模型规模

Kimi Linear v2 报告的模型设置为 48B 总参数、每 token 3B activated parameters。论文脚注说明 activated-parameter 统计不包含 embedding；因此比较模型时应先确认参数计数口径。激活参数较少不表示全部权重都不需要驻留，也不能直接推出训练或推理成本。

论文将 Kimi Linear 建立在 Moonlight 风格的 MoE 主干上。在 1.4T 对照中，MLA、GDN-H 和 Kimi Linear 使用相同参数规模与训练设置；报告给出的 MoE 设置是 256 个专家中激活 8 个（其中包含一个 shared expert），第一层使用 dense 层。这里的专家容量改变通道混合方式，不等同于 KDA/MLA 的序列历史路径。

### 76.1.2 每个 block 先做 token mixing，再做 MoE channel mixing

论文 Figure 3 把一个 block 画成两步：先由 KDA 或 MLA 混合 token 信息，再由 MoE 做 channel mixing。Kimi Linear 在 token-mixing 层之间重复下面的周期：

```text
KDA + MoE → KDA + MoE → KDA + MoE → 全局 MLA + MoE → 重复
```

这里的 $3:1$ 指三个 KDA token-mixing layers 后接一个 MLA token-mixing layer。它不是 MoE 的专家比例，也不应和 Transformer block、attention/FFN 子层等其他计数混用。论文把 MLA 层称为 full/global attention 层，KDA 则按 token 顺序递归更新状态。[Kimi Linear v2，§4 和 Figure 3](https://arxiv.org/html/2510.26692v2)

### 76.1.3 KDA、MLA 与 NoPE 改变的对象不同

KDA 把较早 token 的 key-value 关联递归并入每层自己的有限状态；它不为每个旧 token 保留独立的可读地址。全局 MLA 仍按前缀位置执行 attention，但以每 token latent 表示缓存历史，具体 latent 路径见[第 63 章](63-gated-mla门控潜变量注意力.md)。Kimi Linear 报告给所有 full-attention MLA 层使用 NoPE，并把位置感知与近期性建模主要交给 KDA；这是这份模型报告的设计解释，不是 NoPE 或 KDA 的通用定理。[第 64 章](64-nope与隐式顺序.md)讨论 NoPE 的一般性质。

## 76.2 一条远程事实怎样经过四层周期

设较早的前缀中出现“服务版本为 v2.1”，接近序列末尾的问题要求输出这个精确版本号。沿一组四个 token-mixing layers 追踪，可以看出混合架构提供了哪些路径，也能看出仅有路径还不足以保证答案正确。

### 76.2.1 三个 KDA 层分别更新自己的递归状态

token 经过第一个 KDA 层时，该层读取并更新自己的递归状态；后续两个 KDA 层也分别维护各自的状态。于是每一层都能把对后续预测有用的历史特征向前传递，但这些 state 并非一个共享的、保留原 token 地址的全模型摘要。即使有限状态持续保留了“出现过版本信息”这一线索，也不等于精确字符串 v2.1 一定没有被衰减或干扰。

KDA 的状态矩阵方向、衰减、预测误差和写入过程见[第 61 章](61-kda递归注意力.md)。本章关注状态路径在模型层间怎样与全局读取相接，不重复推导同一递归式。

### 76.2.2 全局 MLA 层提供按位置读取前缀的入口

进入周期中的 MLA 层后，当前位置的 query 可以与 causal mask 允许的历史表示做全局 attention。该层按 token 保留 MLA latent cache，因此模型仍有位置级的全局读取路径。与 KDA 固定形状的递归 state 相比，这条路径更适合让 query 针对历史位置选择内容；但模型必须形成正确的 query、历史表示和后续使用方式，才能把“可读”变成“答对”。

MLA 的 latent 也不是原文副本。query 能访问某个历史位置，不代表 latent 无损保存了版本号，也不保证多个相似版本之间不会混淆。关于 MLA latent 如何生成、缓存和参与计算，见[第 63 章](63-gated-mla门控潜变量注意力.md)。

### 76.2.3 后续状态继续处理全局读取产生的表示

全局 MLA 输出先经过同一 block 的 MoE channel mixer，然后传给下一个 token-mixing layer。下一个 KDA 层继续把当前表示写入自身递归状态；它没有因此获得 MLA 每个 cache slot 的独立地址。对模型层表的正确读法是：不同模块的输出串接在同一条 hidden-state 路径上，但它们保存历史和访问历史的方式仍然不同。

这个例子只能说明可能的信息流。要证明模型会在某个距离准确回答，还需要按目标距离、干扰类型、输入长度和输出要求测量；不能把“周期中有全局层”直接写成“任意远处信息都能被精确找回”。第 65 章进一步比较不同混合架构如何分配局部、全局和递归路径。

## 76.3 为什么选择每三层 KDA 接一层全局 MLA

纯递归状态能以固定形状处理不断增长的前缀，却必须把历史压进有限状态；全局 MLA 保留按位置读取的入口，但需要随 token 保存 latent。混合设计试图平衡这两种成本和能力。Kimi Linear 的比例选择由它自己的消融支持，而不是由“递归层总比 attention 层便宜”这一句直觉决定。

### 76.3.1 有限状态路径会遇到精确复制和检索压力

如果任务只需累计趋势或持续更新某种状态，递归表示可能够用；如果问题要求逐字复制、区分相似实体的字段，或从很远的位置取回确切字符串，有限状态必须在压缩和保留之间作取舍。KDA 的状态机制与合成任务边界由第 61 章讲解；这里的架构问题是，模型是否需要周期性恢复对历史位置的直接读取。

加入全局层也没有消除容量问题。MLA 对历史表示做了 latent 压缩，模型仍可能把 query 对准干扰项、在 latent 中丢失细节，或未能利用读出的内容。因此，层比应当由不同距离与不同信息类型的任务共同检验。

### 76.3.2 比例消融在被测设置中选择了 3:1

论文 Table 1 比较不同 KDA:MLA 比例的训练和验证困惑度，数字越低越好：

| KDA:MLA | 训练 PPL | 验证 PPL |
| --- | ---: | ---: |
| 0:1 | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| 3:1 | 9.23 | 5.65 |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |

在这组测试比例中，3:1 的验证 PPL 最低；1:1 的验证值接近，而 7:1 虽有相同训练 PPL，验证 PPL 更高。作者据此采用 3:1 作为质量和推理开销之间的配置选择。[Kimi Linear v2，§5.2](https://arxiv.org/html/2510.26692v2)

### 76.3.3 3:1 是模型选择，不是 KDA 的定义

递归方程定义 KDA 的状态如何变化；3:1 说明 Kimi Linear 如何布置不同 token-mixing layers；消融结果则说明作者为何在该实验中挑选这一布置。三者处于不同证据层级。将 3:1 搬到其他模型前，应重新验证模型深度、hidden size、任务分布、上下文长度、训练 token、cache 表示和 kernel，而不是直接沿用这个比例。

## 76.4 1.4T token 对照支持哪些质量判断

论文的 1.4T 对照是阅读 Kimi Linear 架构效果时较有用的一组证据：它让 MLA、GDN-H 与 Kimi Linear 在同一模型规模和训练设置下比较。它仍同时改变 KDA、混合比例和全局 MLA 的位置选择，不能单独识别其中哪一项造成分数差异。

### 76.4.1 三种 token mixer 使用相同模型规模和训练设置

论文 §5.4 报告三类模型使用相同 architecture、parameter count 和 training setup。主预训练使用 4,096-token context、共同的 K2 预训练语料与 1.4T training tokens，采用 MuonClip、WSD 学习率计划、$1.1\times10^{-3}$ 学习率和 32M-token global batch；报告还称它们采用相同的 annealing schedule 和 long-context activation phase。[Kimi Linear v2，§5.4.1](https://arxiv.org/html/2510.26692v2)

这使 1.4T 结果比 Kimi Linear 与不同规模公开模型的跨论文排名更适合比较 token-mixing 方案。不过报告比较的是几项设计组合：KDA 的通道级衰减、KDA/MLA 的层间比例和 MLA 的 NoPE 等一起进入配置。它能支持“此套方案在该对照中表现如何”，不能支持“只因为 KDA 所以成绩提高”。

### 76.4.2 MMLU-Pro 与 RULER 展示两个不同长度切片

| 对照模型 | MMLU-Pro@4k | RULER@128k | 长上下文平均 |
| --- | ---: | ---: | ---: |
| MLA | 47.2 | 81.3 | 52.2 |
| GDN-H | 47.9 | 80.5 | 51.2 |
| Kimi Linear | 51.0 | 84.3 | 54.5 |

MMLU-Pro@4k 取自论文预训练结果，RULER@128k 和长上下文平均取自长上下文比较；三种模型都属于 1.4T 对照。Kimi Linear 在这两个代表切片和长上下文平均上领先，但 Table 5 也有反例：它的 LongBench V2 为 35.0，低于 MLA 的 36.1；Frames 为 58.8，低于 MLA 的 60.5。因此“平均结果领先”不等于每项任务都更好。[Kimi Linear v2，Tables 3–5](https://arxiv.org/html/2510.26692v2)

### 76.4.3 SFT 分数需与预训练阶段分开阅读

论文 Table 4 汇报相同 SFT recipe 下的 instruction-tuned 结果。部分指标领先并不意味着全部指标都领先：例如 LiveBench (Pass@1) 上 Kimi Linear 为 45.2，低于 GDN-H 的 46.4；EvalPlus 上 Kimi Linear 为 61.0，低于 MLA 的 62.6 与 GDN-H 的 62.5。预训练、SFT 和 RL 是不同阶段与目标，不能把这些分数混在一起当作同一评测排行。

## 76.5 5.7T 发布 checkpoint 回答的是另一类问题

1.4T 结果用于同设置的模型比较；最终发布 checkpoint 则继续训练到 5.7T tokens，并支持最多 1M context。比较这两类结果时，除了结构，还改变了训练量与 checkpoint 阶段。模型发布成绩不能自动解释成某一层比或单个机制的因果收益。

### 76.5.1 1M context 是 checkpoint 支持和特定评测，不是所有任务的保证

论文说明最终 Kimi Linear checkpoint 使用 5.7T tokens 训练，并支持最长 1M tokens。Appendix D 的 instruction-tuned Kimi Linear 在 RULER@1M 得分为 94.8。这个数字证明报告中的 checkpoint 在指定 RULER 配置下取得该成绩；它不等于对所有任务、所有信息位置、任意输入质量的长文本理解保证。

### 76.5.2 与 Moonlight 的 5.7T 比较不是同总参数消融

Appendix D 的 Kimi Linear Base 与 Moonlight Base 都使用 5.7T training tokens，二者 activated parameters 都为 3B；但 Kimi Linear 的 total parameters 是 48B，Moonlight 为 16B。这个比较可展示两个发布模型在报告任务上的结果，却不能隔离注意力结构的贡献，因为总参数规模也不同。

### 76.5.3 1M RULER 指标不能替代匹配的全注意力基线

Instruction 结果还存在上下文范围差异：论文说明 Moonlight-Instruct 没有评测超过 8K context 的任务，因此 Kimi Linear 的 RULER@1M 没有同表的长上下文 Moonlight 基线。不能把它改写成“在 1M 上优于全注意力模型”。[Kimi Linear v2，Appendix D](https://arxiv.org/html/2510.26692v2)

## 76.6 分开核算 cache、prefill 与 decode 效率

### 76.6.1 3:1 层表解释 KV cache 主张，但不等于总显存比例

若只按 token-mixing 层数比较，四层周期中有三层 KDA、一层全局 MLA。相对于四层都使用全局 attention 的同层数结构，Kimi Linear 只有约四分之一层保留逐 token 的 MLA latent cache。这提供了论文“KV cache 使用最多减少 75%”主张的结构直觉。

这个比例不表示整模型的所有历史状态减少 75%。KDA 仍需保存每层递归状态和短卷积状态；MLA 的 latent 宽度不等于标准多头 K/V；运行时还可能包含元数据、对齐和工作区。第 69 章说明如何逐层核算显式 KV、latent 和递归 state，估算时不应把这些对象混为同一 cache。

### 76.6.2 Figure 7 区分 prefill latency 和单 token decode

论文 Figure 7 使用 Kimi Linear 48B setting，比较模型具有相同层数和 attention heads，图注注明 batch size 为 1。报告称在 512k 输入长度下，Kimi Linear 的 prefill 约快 2.3 倍；在 1M 下约快 2.9 倍。这里测的是 prefill latency，不能当作每 token decode 时间或任意并发量下的吞吐。[Kimi Linear v2，§5.6 与 Figure 7](https://arxiv.org/html/2510.26692v2)

### 76.6.3 Figure 1 的 6.3× TPOT 依赖另一种 batch 口径

论文 Figure 1(b) 在 1M 场景报告 6.3× TPOT（1.84 ms 对 11.48 ms），并将其与较低内存占用后可以使用更大 batch 联系起来。它与 Figure 7 图注中的 batch-one 曲线不是同一测量条件；不能把 6.3× 写成单请求、任意硬件都能复现的固定加速。Figure 1(a) 的 RULER@128k 3.98× acceleration 又是另一项图示比较，也不等于 TPOT 或 prefill 加速。

论文没有在相关图中提供复现实测服务所需的全部硬件、驱动、runtime、kernel 和调度信息。数字应保留“论文报告”“指定序列长度和 batch 条件”等限定；生产系统仍需在目标实现上单独测量。

## 76.7 用证据而不是层比回答“混合架构是否更好”

阅读新的 hybrid 模型时，可以先把问题拆成三栏：一栏记录历史表示和可访问路径；一栏记录质量评测的模型阶段、长度与任务；一栏记录 cache、prefill、decode、batch 和硬件条件。Kimi Linear 案例展示了为何这几栏不能互相替代：3:1 说明模型层表，1.4T 结果回答报告内匹配模型对照，5.7T 结果描述发布 checkpoint，而 6.3× 是特定更大 batch 条件下的 TPOT 主张。

### 76.7.1 设计评测时按信息访问需求和上下文长度分桶

如果目标是精确复制，应测量目标位置与干扰项的距离、字符串/字段是否完全匹配、错误位置是否被召回；如果目标是长生成，则要分开记录输入长度、输出长度、batch、TPOT 和缓存字节。

### 76.7.2 只有匹配模型和执行条件才能比较质量与速度

再用相同模型规模、数据、training tokens、后训练阶段和 kernel 逐项控制变化。若模型一次同时改变 KDA、MLA、NoPE、层比和训练配方，结论应落在整套配置上。

### 76.7.3 Kimi Linear 是一个具体实现案例，不是架构配比处方

本章讨论 Kimi Linear 一个特定版本。想追问 KDA 的状态递推，回到[第 61 章](61-kda递归注意力.md)；想看全局 MLA latent cache，读[第 63 章](63-gated-mla门控潜变量注意力.md)；NoPE 的位置边界见[第 64 章](64-nope与隐式顺序.md)；混合路径横向比较见[第 65 章](65-hybrid-attention局部全局与递归状态.md)；Kimi K3 的模型结构和版本差异见[第 88 章](88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)。记住：递归状态与全局位置读取可以互补，但模型有没有学会在任务中正确使用它们，仍要由匹配条件下的评测回答。
