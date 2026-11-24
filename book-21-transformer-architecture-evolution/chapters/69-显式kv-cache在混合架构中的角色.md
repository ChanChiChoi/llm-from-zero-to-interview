# 第 69 章 显式 KV Cache 在混合架构中的角色

混合架构把不同的历史处理路径放进同一个模型：有的层按 token 保存 key/value（KV），有的层只看局部窗口，有的层把每个位置压进较窄的 latent，还有的层递归更新固定形状的 state。这里的“cache”因此不是一种统一的数据结构。判断一个模型是否减少了显式 KV，要逐层确认它保存什么、能读取哪些位置，以及这些表示怎样随上下文变长。

本章以前置的[第 11 章：KV Cache 长上下文推理和显存增长问题](11-kv-cache长上下文推理和显存增长问题.md)为基线，重点讨论混合层表怎样映射为逻辑 payload。第[65 章](65-hybrid-attention局部全局与递归状态.md)比较序列路径；第[63 章](63-gated-mla门控潜变量注意力.md)与第[70 章](70-latent-cache压缩历史表示.md)分别承担 MLA 和 latent cache；第[71 章](71-recursive-state-cache递归状态缓存.md)讲递归状态生命周期。分页布局和 serving 调度留给[第 20 章](20-FlashAttention与PagedAttention.md)及[第二十四册](../../book-24-llm-inference-engine/目录.md)。

## 69.1 混合模型中的 KV 是按注意力路径保留的状态

理解混合模型的 cache，先把“层做什么”与“层留下什么”分开。注意力路径可能保留逐 token 的表示，递归路径则在每个新 token 到来时更新状态；它们都延续了历史信息，却提供不同的读取方式，也有不同的容量增长曲线。

### 69.1.1 全局 attention 为保留位置写入 K/V

在标准 causal attention 层，输入位置经投影得到 key 和 value。Prefill 阶段一次处理提示词并生成这些表示；decode 阶段每来一个新 token，只需为新位置追加 K/V。后续位置的 query 会对允许访问的历史 key 计算权重，再从对应 value 中取出加权内容。

例如，第五个 token 到来时，该层可以用自己的 query 比较前四个位置的 keys，并按所得权重组合四个 values。缓存保留了“哪个位置对应哪组 K/V”，所以该层具备按位置直接读取历史的路径。它并不保证 query 一定选中正确证据，也不保证模型会正确解释读到的内容；这些属于模型能力和任务表现，不能仅凭 cache 结构推出。

通常每层分别产生自己的 K/V，因此普通全局 attention 的 payload 随保留 token 数增长。第 11 章给出单层与整模型的基础估算；混合模型的关键变化是只有一部分层走这条路径，且不同层的 head 数、维度和保留长度可能不同。

### 69.1.2 局部 KV、latent cache 与递归 state 代表不同历史

不同路径都可能被统称为 cache，但它们保留的信息形式并不相同：

| 历史表示 | 每个位置留下什么 | 后续读取方式 | 典型增长方式 |
| --- | --- | --- | --- |
| 全局显式 K/V | 每个可访问 token 的 key 和 value | Query 可按内容读取整个保留前缀 | 随保留长度增长 |
| 局部窗口 K/V | 可见窗口内各 token 的 key 和 value | 只能直接读取本层窗口内的位置 | 窗口固定时受窗口大小限制 |
| 逐 token latent | 每个 token 的压缩 latent，可能由它重建 attention 所需表示 | 仍可按位置进行 attention，但使用压缩表示 | 随 token 数增长，单位置 payload 改变 |
| 递归 state | 输入不断更新的固定形状或有限组状态 | Query 读取累计状态，通常不能直接寻址任一原始位置 | 对递归模型通常不随前缀长度线性增长 |

局部窗口的“少存”来自保留范围受限；latent cache 改变每个位置的表示宽度；递归路径把历史持续归并到 state 中。三者不是同一种压缩倍率。MLA 的 latent 细节见[第 70 章](70-latent-cache压缩历史表示.md)，递归 state 的更新和恢复见[第 71 章](71-recursive-state-cache递归状态缓存.md)。

### 69.1.3 层表需要记录表示类型和可见历史

只读到模型使用“attention”或“hybrid”仍不足以估算缓存。逐层审阅时至少要确定混合器类型、该层可见的历史范围、保存的是普通 K/V 还是 latent/state、每位置或每层的状态形状，以及长度增加时保留量如何变化。若报告或配置没有披露某项，就把它标成未知，不能按同家族其他模型补猜。

一个简化的层表可以写成这样：

| 层路径 | 可见范围 | 需要核对的持久状态 | 主要容量变量 |
| --- | --- | --- | --- |
| Full attention | 全部 causal 前缀 | 逐 token 的 K/V | 保留 token、KV heads、head dimension、dtype |
| Sliding-window attention | 最近窗口 | 窗口内逐 token 的 K/V | 窗口长度及每层 KV 形状 |
| GDN/KDA 等递归路径 | 历史被并入递归更新 | 每层或每头的递归 state | 状态形状、精度和请求数 |
| MLA 类路径 | 可跨全前缀读取 | 逐 token latent 及必要的附加表示 | token 数、latent 维度及附加项 |

这张表记录的是模型计算所需的逻辑状态。实际运行时的物理布局、分页与调度由推理引擎决定；两者不能混为一张容量表。

## 69.2 混合架构选择显式路径的位置与形式

“混合”描述的是若干计算路径如何共存，不预先规定它们的缓存格式。层间交错、同层并行、注意力可见范围和跨层共享是不同设计轴；分析时应分别回答“哪些层读取哪些历史”和“系统保留了哪些表示”。

### 69.2.1 层间交错只让部分层增长为显式历史

Qwen3.8-Next 技术报告描述的基础架构每四层安排一层 global attention，其余三层使用 Gated DeltaNet（GDN）；报告还说明，在 continued pretraining 阶段将这些 full-attention 层替换成 Qwen Sparse Attention（QSA）。这是特定模型与训练阶段的安排，不能简化成所有阶段都固定使用同一张层表。[报告](https://arxiv.org/html/2608.30320v1)把 GDN 的递归状态与 attention 的逐 token 读取作为不同路径讨论；其中 QSA 的索引会挑选供稀疏 attention 计算的位置，但这件事本身并没有说明完整 KV payload 会按什么策略保存或淘汰。

Kimi K3 报告则描述了每三个 Kimi Delta Attention（KDA）层后接一层 Gated MLA，并在主干末端再放置 Gated MLA。这里 KDA 使用递归状态，Gated MLA 为 token 位置保留低维 latent 并重建 attention 所需内容；它不是普通的每头完整 K/V。[Kimi K3 报告](https://arxiv.org/html/2607.24653v2)提供的是该模型的具体布局，不是所有混合架构的推荐比例。两例共同说明：层数比例只能告诉我们路径放在哪里，容量仍要按实际表示核算。

### 69.2.2 层内 hybrid heads 与跨层 KV sharing 改变不同成本

Hymba 采用同一层内并行的 attention heads 与 state-space heads。并行分支让该层同时使用可寻址的注意力路径与状态路径；论文还分别研究局部/全局 attention 和相邻层 KV sharing。前者改变注意力的可见范围，后者改变重复保存的表示数量，不能把它们都概括成“多一个 state 分支”。

[Hymba 论文](https://arxiv.org/html/2411.13676)报告的质量、cache 和吞吐观察来自其小模型、训练配方和评测设置。若一组实验同时加入 hybrid heads、调整窗口并共享 KV，就只能把结果归于该组合，不能单独证明某个设计带来了全部收益。跨层共享还必须确认被共享的 K/V 在投影、位置处理和运行时布局上确实可复用。

### 69.2.3 全局、窗口与稀疏路径改变的不是同一个量

全局、窗口和稀疏 attention 首先描述 query 可以读取哪些 token；普通 K/V、latent 与递归 state 则描述历史怎样保存。两组概念有关联，但不能互相替代：

| 设计变化 | 直接改变 | 不能单独推出 |
| --- | --- | --- |
| 全局 attention 改为滑动窗口 | 本层可见位置；标准窗口实现可将保留长度限制在窗口内 | 窗口外信息对整模完全不可达，或其他层不会传递它 |
| Dense attention 改为 sparse attention | 每个 query 实际参与核心 attention 计算的位置集合 | 主 KV cache 必定只保存被当前 query 选中的位置 |
| 普通 K/V 改为 latent | 每个位置保存的表示及其维度 | cache 不再随 token 数增长，或 latent 无损保留所有细节 |
| Attention 层改为递归路径 | 历史的更新与访问方式 | 固定状态一定能完成每种逐位置检索任务 |
| 跨层 KV sharing | 多层间是否重复保存可复用表示 | 任意相邻层的 K/V 都可以安全共享 |

以 QSA 为例，Qwen 报告说明索引器先对 micro-block 打分，再为每个 query 选出 token 位置供核心稀疏 attention 使用；报告还分别分析了索引与 attention 计算开销。因为所选位置会随 query 变化，不能由“稀疏读取”直接断定服务端可丢弃其余 token 的主 K/V。若实现确实只保留部分主 K/V，还需要报告或固定版本实现说明它如何应对后续 query 选中其他位置、如何恢复遗漏表示。本章不把这一未披露策略写成架构必然行为。

## 69.3 按层计算显式 KV payload

架构层表可以转成普通逐 token K/V 的逻辑 payload 估算。这个公式只回答“按声明的层和形状，需要保存多少 K/V 元素”；它不包含递归状态、latent、权重、临时 workspace 或分页元数据，也不直接等于设备实际显存。

### 69.3.1 全局 K/V 层按保留 token 数逐层求和

对每层独立存储普通 K/V 的层集合 $\mathcal{A}_{\mathrm{KV}}$，令 $B$ 为批内请求数，$T_{i,\ell}$ 为请求 $i$ 在第 $\ell$ 层实际保留的位置数，$H_{\mathrm{kv},\ell}$ 为该层 KV head 数，$d_{h,\ell}$ 为每个 head 的维度，$b_{\ell}$ 为每个元素的字节数。K 与 V 各保存一份，因此批内普通 KV payload 可估为：

```math
M_{\mathrm{KV,payload}}
\approx
2\sum_{i=1}^{B}\sum_{\ell\in\mathcal{A}_{\mathrm{KV}}}
T_{i,\ell}H_{\mathrm{kv},\ell}d_{h,\ell}b_{\ell}.
```

全局层通常让 $T_{i,\ell}$ 等于当前保留的 causal 前缀长度；标准滑动窗口层可简化为该前缀长度与窗口 $w_{\ell}$ 的较小值。批内请求长度不同时，应逐请求相加，不能用最大上下文长度乘整个 batch 来冒充实际利用率，也不能忘记 padding 或运行时分配粒度可能带来的额外占用。

集合 $\mathcal{A}_{\mathrm{KV}}$ 只包含独立保存普通显式 K/V 的层。若模型跨层共享同一份 K/V，应按唯一存储项核算，而不是按引用它的层数重复相加；若要估算单卡占用，还须结合张量并行如何切分 heads 和层。这个口径与[第 11 章](11-kv-cache长上下文推理和显存增长问题.md)的整模型基线一致，只是把求和限制到真实使用普通 K/V 的层。

### 69.3.2 窗口、GQA、latent 与 state 分别改动账本字段

读公式时可把每种设计对应到具体字段。窗口改变 $T_{i,\ell}$；Grouped-Query Attention（GQA）或 Multi-Query Attention（MQA）改变 $H_{\mathrm{kv},\ell}$；KV 量化改变 $b_{\ell}$；跨层共享改变实际存储的独立副本数。每一种变化影响的字段不同，必须查本模型配置与实现，不能拿“global 层占比”代替它们。

MLA 类 latent cache 仍可能按 token 保存表示，但每个位置存的是 latent 及模型规定的附加信息，应另建一项账本，不能把其维度直接填进普通 K/V head 数。递归 state 则按每请求、每层的状态张量形状和 dtype 单独记账；固定状态大小表示它不随历史 token 数线性增长，不代表状态开销为零。分页块、scale、对齐、索引器和 workspace 属于实现或辅助状态，也应按各自定义统计，而非混入普通 K/V 公式。

### 69.3.3 少数 global 层的比例只在等形状假设下成立

看一个只用于核算比例的例子：单条请求有 $32$ 层，每层形状相同，$H_{\mathrm{kv}}=8$、$d_h=128$，使用 BF16（每元素 $2$ bytes），保留 $8192$ 个 token。若所有层都存普通全前缀 K/V，payload 为 $1$ GiB；若只有 $4$ 层如此，其余层的状态先不计，普通 K/V payload 为 $128$ MiB，也就是全层基线的 $1/8$；若有 $8$ 层，则为 $256$ MiB，即 $1/4$。这只是同形状、同长度、同精度条件下的比例演示。

真实模型中，少数 global 层可能有不同的 KV head 数、head dimension 或精度；窗口层也会留下自己的窗口 K/V，latent 与递归路径各有 payload。因而 $4/32$ 不等于整模型总状态就是全模型的八分之一，更不能推出实际显存或推理时间也降低到八分之一。容量管理和设备占用的完整分析见[第二十四册 KV Cache 章节](../../book-24-llm-inference-engine/目录.md)。

## 69.4 评估保留路径的质量与成本

显式逐位置读取可以提供一种访问路径，是否有助于目标任务仍要测量。合适的评估既要看模型找不找得到目标信息，也要把层位置、表示形式、计算预算和服务环境记录下来，避免从单一 benchmark 推出普遍架构结论。

### 69.4.1 检索与复制任务应针对目标信息粒度

若要判断哪条路径保留了什么能力，测试集应覆盖不同的信息粒度：精确复制编号或字符串、远距离事实检索、跨段组合、多个相似候选之间消歧、冲突版本判断，以及一般语言任务。精确匹配、答案级准确率和证据位置命中率回答的是不同问题；对只检查最终答案的 benchmark，不能直接声称模型确实读到了正确原文位置。

已有论文展示了这种任务条件差异。[Griffin](https://arxiv.org/html/2402.19427v1)分别研究 synthetic copy/retrieval 与 phonebook lookup，并把 full-attention KV、local cache 和 recurrent state 的容量分开核算；[Hymba](https://arxiv.org/html/2411.13676)报告了其架构下的 recall 实验；[Qwen3.8-Next](https://arxiv.org/html/2608.30320v1)则比较 QSA 与 full-attention 配置在 RULER 和 8-needle MRCR 等长上下文任务上的表现。三者使用的模型、训练预算、数据和评测设置不同，不能把分数直接横向比较，也不能由某一组结果推出所有递归模型都会丢失远程细节。

### 69.4.2 消融应隔离层位置、数量与表示方式

要回答“多少显式层是否足够”，实验至少要分开改变显式层的数量与位置、注意力范围（global、window 或 sparse）、保存形式（普通 K/V 或 latent）以及递归路径。层的位置也可能重要：前、中、后层承担的变换不同，所以只比较层数不能替代位置消融。

比较时应尽量对齐模型规模、训练数据与 token 数、上下文长度、优化设置、评测集和训练/推理预算。若替换路径同时改变参数量、计算量或训练成本，就记录这些差异，按质量—资源曲线比较；若多个组件一起变化，只能报告联合结果。Qwen 报告明确提醒，其 GDN hybrid 架构消融不能隔离所有组件的因果贡献；同样的审慎也适用于其它混合模型。需要看懂 GDN、KDA 或 MLA 内部机制时，分别回到[第 65 章](65-hybrid-attention局部全局与递归状态.md)、[第 71 章](71-recursive-state-cache递归状态缓存.md)和[第 70 章](70-latent-cache压缩历史表示.md)。

### 69.4.3 报告质量、payload 与系统成本的条件

一份可复核的比较应同时给出：各层采用的表示和可见范围；普通 K/V 与 latent/state 的逻辑 payload 随上下文长度如何变化；复制、检索和常规任务的质量指标；以及 prefill、decode 的测量条件。若报告系统延迟或吞吐，还需注明模型版本、硬件、kernel、dtype、batch/并发和计时口径。

逻辑 payload 不是整机显存；实际占用还受分页粒度、对齐、索引器、临时 buffer、并发请求和运行时分配影响。理论上读更少的 K/V 也不自动带来更低延迟，实际收益取决于索引开销、内存布局和 kernel。第 20 章讲 FlashAttention 与 PagedAttention，第六册和第二十四册讲推理服务及性能验收；本章只把架构路径和逻辑状态账本说清。结论应落在具体模型、任务和测量条件上：显式 attention 提供逐位置读取的机制，latent 与递归路径改变历史表示，哪种组合值得保留由质量和资源证据共同决定。
