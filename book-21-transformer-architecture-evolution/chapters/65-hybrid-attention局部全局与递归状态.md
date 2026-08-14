# 第 65 章 Hybrid Attention：局部、全局与递归状态如何分工

## 65.1 为什么不让每一层都做 full attention

full attention 让每个 query 访问完整历史，表达力强，适合精确引用和跨段对齐；但它的计算、KV cache 和带宽都随上下文长度增长。局部 attention 把可见范围限制在窗口，递归路径把历史压缩成 state，全局层则保留少量远距离读取能力。

Hybrid attention 把这几条路径放在同一个模型里。它不是简单的“省显存版本”，而是把不同类型的依赖分配给不同层：邻近语法不必经过全局访问，重复流可以由状态扫描，真正需要精确跳转的关系交给 global attention。

## 65.2 三种路径的角色

local attention 适合相邻 token、代码局部结构和短距离实体；global attention 适合跨段证据、版本比较和精确引用；recursive/state path 适合流式输入、重复模式和固定大小的历史摘要。

第 `l` 层可以选择路径：

```math
p_l\in\{\mathrm{local},\mathrm{global},\mathrm{recursive}\}
```

层表 `p_1,\ldots,p_L` 决定模型的表达路径和部署成本。即使 global 层很少，只要它们保存完整历史，长请求的显式 KV 仍可能是主要资源。

## 65.3 窗口、全局和 state 的连接

局部层的窗口为 `w` 时，位置 `t` 只访问 `[t-w+1,t]`。递归层将更早的内容更新成 `S_t`。全局层可以直接访问选定历史，或者读取被压缩的 global summary。

一个抽象的混合输出为：

```math
y_t=\alpha_t y_t^{\mathrm{local}}
+\beta_t y_t^{\mathrm{global}}
+\gamma_t y_t^{\mathrm{state}}
```

其中系数可以是固定 schedule，也可以由门控学习。真实实现未必显式相加，但这个表达能帮助理解：路径之间存在表达力和资源分工。

## 65.4 一个百万 token 日志例子

一百万 token 的日志中，99% 是重复心跳，1% 是异常堆栈和版本变更。local 层能处理每条日志附近的语法，recursive 层能压缩重复模式，global 层负责把异常堆栈与开头的发布记录对齐。

如果没有 global 层，模型可能知道“有异常”，却无法引用最初的版本号；如果所有层都 global，质量可能更稳，但 KV、prefill 和并发成本增加。这个例子说明 hybrid 的核心不是平均地省资源，而是把昂贵路径留给真正需要远距离精确交互的部分。

## 65.5 schedule 的 trade-off

增加 global 层通常提高远程检索能力和显存成本；减小 local window 降低计算，却可能破坏代码括号和局部语义；增加 recursive 层使长流更便宜，却增加压缩误差。层的位置也重要：早期 global 层、晚期 global 层和均匀间隔的效果可能不同。

因此模型卡中的“某比例 attention 层”不能直接转成准确的服务并发。还要知道窗口大小、KV dtype、latent/state 维度、kernel、batch 和 scheduler。

## 65.6 训练中的混合边界

训练数据要让每条路径学到它负责的任务。若 global 层很少而训练中没有跨段证据任务，模型可能只把它们当普通层使用；若 local window 与训练/推理不一致，窗口边界可能产生回归；若递归 state 只在短 chunk 中重置，长流服务可能出现未训练的状态漂移。

长上下文继续训练应按长度、任务和证据位置分桶，同时测短任务回归。单看总 loss 不能知道是哪条路径在承担能力。

## 65.7 Serving 的状态布局

请求状态至少有三类：local KV、global KV 或 latent cache、recursive state。allocator 要按层表为每种对象分配 block；scheduler 要知道请求增长时哪一类状态增加；完成、取消、preemption 和 prefix sharing 也要分别处理。

如果只实现了 KV release，却忘记释放 recursive state，长时间运行会出现隐蔽泄漏；如果 batch 重排改变了 layer schedule 或 state index，结果会错误但不一定 OOM。

## 65.8 评估不能只跑平均 perplexity

最小评测包含局部复制、跨窗口引用、重复日志、精确数字、冲突版本、长程引用和工具参数。对每个任务记录答案准确率、证据召回、引用支持、长度曲线、state/KV GiB、TTFT、TPOT、p99 和并发。

做路径消融：全 global、全 local、全 recursive、hybrid。再做层 schedule 消融和窗口消融。只有这样才能判断 hybrid 的收益来自路径组合，还是来自额外参数和训练预算。

## 65.9 一个调度故障

线上短请求 p99 突然升高，长请求数量没有明显增加。检查发现 scheduler 为每个长请求预留了全部 global KV，即使实际只到达了部分长度；同时 recursive state 不能被抢占，短请求无法插队。修复需要让 global KV 按需增长、为 state 定义 snapshot，或者把长请求路由到专用池。

这个故障说明架构图和部署图必须一致。模型的 hybrid 设计如果没有对应的 allocator 和 scheduler，就只能停留在论文图里。

## 65.10 常见失败

把 local attention 当作无损压缩；从路径比例推断完整能力；global 层的 KV 没有计入容量；窗口边界 off-by-one；递归 state 没有 reset；只测短上下文；忽略 hybrid kernel fallback；用一个平均速度掩盖证据召回下降。

## 65.11 面试回答与练习

回答“为什么要 hybrid attention”时，应说局部、全局和递归路径承担不同距离的依赖：local 控制短距离成本，recursive 压缩长流，global 保留精确远程读取。设计的难点是层 schedule、信息瓶颈、KV/state 生命周期、kernel 和评估，不能只比较理论复杂度。

练习一：为 24 层模型设计三种 layer schedule，说明每种适合的任务。

练习二：估算 4 个 global 层仍然可能造成的 KV 压力。

练习三：设计一个检测 local window 边界错误的任务。

### 65.11.1 三条路径的职责

local attention 擅长短距离语法和局部模式，global attention 提供精确的远程 token 读取，recursive state 维护低成本的历史摘要。hybrid 设计不是把三者随机交替，而是决定哪些层、哪些 head 和哪些 token 需要哪一种信息通路。

一个 24 层教学 schedule 可以是 16 层 local、4 层 recursive、4 层 global。它可能适合长文档的总体建模，但如果四层 global 共享全部序列，KV 仍然很大；如果 global 位置错误，局部层和递归层无法弥补精确引用损失。

### 65.11.2 窗口边界和信息接力

local window 的边界会切断相邻块之间的直接访问。系统可以使用重叠窗口、跨块摘要、周期性 global 层或递归 state 传递信息。每种方法都引入不同的误差：重叠增加 token 和计算，摘要有压缩损失，global 增加 KV，递归 state 难以精确引用。

检测边界错误的测试应把一条关键事实放在窗口边缘两侧，改变块对齐方式并要求引用。若只在某个对齐下成功，问题可能是窗口或 position reset，而不是模型知识。

### 65.11.3 调度的部署含义

不同层使用不同 cache/state，allocator 不能只按“每层一个 KV block”处理。batch 重排、preemption、prefix sharing 和 checkpoint 都要知道每条路径的状态类型。调度器若只回收显式 KV，递归或 latent state 可能继续占用显存；若只复制 KV，不复制 state，恢复结果会不一致。

## 65.12 schedule 不是越复杂越好

混合层的 schedule 需要同时考虑信息流和硬件形状。若每一层都在 local、global、recursive 三条路径之间切换，模型表达空间可能更大，但 kernel、cache 类型和状态管理也更复杂。一个固定的周期 schedule 往往更容易训练和部署；动态 schedule 只有在任务收益足够大时才值得。

可以先从三种基线开始：全 global、全 local/recursive、固定比例 hybrid。只有 hybrid 在精确引用、长程冲突和单位成功成本上形成稳定优势，才继续尝试 token 级 gate 或任务级路由。

## 65.13 global 层应该放在哪里

global 层放在浅层，可能帮助早期建立跨段连接；放在深层，可能让已经提取的局部特征完成全局组合。没有普适答案，层位置必须用消融确定。

设 global 层集合为 L_g，显式 cache 的粗略 payload 受其影响：

~~~math
M_{\mathrm{global\_KV}}
\propto |L_g|BT H_{\mathrm{kv}}d_hb
~~~

但不同层可能有不同 head、窗口和压缩策略，所以这个式子只用于预算估算。实验应同时改变 global 层数量和位置，观察长程 recall、局部 loss、TTFT 和 decode。

## 65.14 混合训练中的路由信号

如果训练时 global 路径总是可用，模型可能懒得学习递归压缩；如果 global 路径过早关闭，模型又可能学不到精确检索。可以采用 curriculum：先训练完整路径建立任务能力，再逐步增加 local/recursive 比例，最后用固定路径和真实 serving kernel 验证。

路由损失不应只鼓励稀疏。还要惩罚关键证据任务中的漏读，并用路径消融检查模型是否真的依赖了 global 层。一个 gate histogram 好看，不等于任务分工正确。

## 65.15 混合架构的端到端回退

当 state norm 异常、global cache 不足或压缩证据置信度低时，系统需要回退。回退可以是增加 global layer、重新 prefill、调用 retrieval 或转人工。回退路径应在 trace 中记录原因、额外成本和最终质量。

如果系统没有回退，只是在平均 benchmark 上节省资源，生产中一个关键长程失败可能抵消所有收益。高风险请求应使用更保守的 schedule。

## 65.16 从结构直觉走向容量账本

Hybrid attention 是能力路径和资源路径的联合设计。它可以让大多数历史处理更便宜，同时保留少量精确通道，但必须用任务分桶、状态容量和端到端服务指标证明 trade-off 成立。

公开模型卡和论文可以支持路径方向；具体层 schedule、窗口、state layout 和 runtime 行为以对应实现为准。

## 65.17 三条路径分别保存什么

混合注意力不是把几个模块并排放在图上，而是让不同模块承担不同的历史表示任务。可以先用三个抽象来区分：

1. local attention 保存最近窗口内的显式 token 关系，擅长局部语法、短距离复制和邻近代码依赖。
2. global attention 或稀疏全局路径保留跨段的直接寻址能力，擅长远程实体和冲突版本。
3. recursive state 把历史压缩为有限状态，擅长流式扫描、趋势和重复模式。

如果把它们都称为“记忆”，读者会错过最重要的差别：显式路径有 token 地址，窗口路径有有限地址，递归状态通常只有聚合地址。一个系统是否适合审计，不取决于它是否有长上下文窗口，而取决于关键事实能否从某条路径回读并带上来源。

## 65.18 计算账本不能只数层数

设模型有 N 层，其中 full/global attention 占比例 p_g，local attention 占比例 p_l，recursive state 占比例 p_s，三者之和为 1。一个粗略的成本模型是：

~~~math
C_{\mathrm{layer}}
=N\left(
p_g C_{\mathrm{global}}
+p_l C_{\mathrm{local}}
+p_s C_{\mathrm{state}}
\right).
~~~

当序列长度为 L 时，global 路径可能呈二次或稀疏化后的次二次成本，local 路径近似与窗口 w 成正比，state 路径近似与固定 state size 成正比。这个式子只是第一步，实际还要加入：

~~~math
C_{\mathrm{total}}
=C_{\mathrm{FLOPs}}
+C_{\mathrm{memory\ traffic}}
+C_{\mathrm{communication}}
+C_{\mathrm{launch}}.
~~~

例如把 80% 的层换成递归 state，不代表总成本减少 80%。如果剩余 20% global layer 需要跨 GPU 通信，或 state kernel 每 token 产生大量不连续读写，通信和带宽可能成为新瓶颈。

## 65.19 schedule 是模型能力假设

常见 schedule 有三种：

| schedule | 直觉 | 风险 |
| --- | --- | --- |
| 周期式 global | 每隔若干层提供一次远程通道 | 远程证据可能在两次 global 之间被压缩 |
| 分段式 global | 在某些深层集中做跨段融合 | 前层产生的局部表示可能已经丢失关键细节 |
| 内容路由式 global | 难样本或特定 token 才打开全局路径 | 路由误判会造成不可预测的长程失败 |

周期式最容易实现和压测，内容路由式最有潜在节省，但也最难验证。路由器需要知道“哪些 token 需要全局访问”，而这本身就是一个轻量的检索问题。若只按 token 类型、标点或长度做启发式路由，可能把真正的关键事实误判为普通文本。

## 65.20 长程证据如何穿过混合层

假设证据在位置 100，问题在位置 100000。若中间只有递归层和 local window，证据必须通过 state 压缩或逐层残差传递到末端；若某层有 global attention，末端 query 可以直接访问该层保留的显式表示。两者的错误形态不同：

1. state 路径可能把数字、否定和版本条件压成模糊主题。
2. local 路径可能根本看不到远处证据。
3. global 路径可能找到证据，但被错误的 query 或后续融合淹没。
4. serving 可能有理论上的 global path，却因为窗口、page 或 mask 配置没有真正开放。

因此长上下文测试要保存 path trace，至少知道答案使用了哪类路径。只看最终文本无法判断失败来自检索、压缩还是融合。

## 65.21 混合层的状态隔离

每条路径都有自己的边界条件。local attention 需要 window offset，global attention 需要 block/page metadata，recursive state 需要 reset 和 snapshot。一个请求从 prefill 进入 decode，再经历 preemption、迁移和恢复时，三类状态必须一起推进：

~~~text
logical position
local window offset
global page mapping
recursive state
model revision
~~~

如果只恢复 KV 而没有恢复 state，模型可能在普通问答上继续生成，却在跨段任务上突然失忆；如果只恢复 state 而没有恢复 page offset，global path 可能读到错误位置。恢复测试必须比较不间断运行和保存/恢复运行的逐 token logits。

## 65.22 一个层 schedule 的数值例子

设 24 层模型采用 4 层 local/state 后接 1 层 global 的周期模式，则每 5 层有 1 层 global，global 比例约为 20%。若单层 global 的长上下文成本是 10 个单位，local/state 平均是 2 个单位，粗略层成本是：

~~~math
C=0.2\times10+0.8\times2=3.6.
~~~

全部 global 的成本是 10，理论上节省约 64%。但若 global layer 的引用支持率从 96% 降到 82%，这个 trade-off 可能不适合高风险任务。还要计算 batch、通信和 verifier 的额外成本。真正的目标不是最小化 C，而是在质量门槛下最小化单位成功任务成本。

## 65.23 与 RAG 的组合

混合 attention 不能替代外部检索。一个实用架构是：

1. recursive/state 路径处理对话和文档背景。
2. retriever 根据 query 找候选原文。
3. local/global attention 精读候选片段。
4. verifier 检查引用、数字和版本。

这样模型不必让 state 永久保存所有原文地址；外部索引负责可追溯，模型负责理解和组合。代价是多了一次检索延迟、权限过滤和文档版本治理。对企业系统而言，这通常比把所有审计责任交给隐式状态更可控。

## 65.24 混合注意力的消融实验

至少做四个 baseline：

| baseline | 用途 |
| --- | --- |
| all-global | 能力上限和资源上限 |
| all-local/state | 成本下限和长程失败参考 |
| fixed hybrid | 观察固定 schedule 的折中 |
| routed hybrid | 观察动态路由是否真正节省 |

在同一训练 token、参数量、tokenizer 和上下文长度下，测试局部语法、远程 needle、多证据组合、冲突版本、代码跨文件和工具参数。每个任务再按证据位置分桶，避免平均值掩盖中间位置失败。

## 65.25 线上路由和尾延迟

动态打开 global path 会产生请求之间的计算差异。若少数难请求使用更多 global layer，平均延迟可能下降但 P99 上升；连续 batching 还可能因为不同路径 shape 产生 batch 分裂。需要记录：

~~~text
path activation rate
tokens per path
batch split count
queue time
TTFT
TPOT
P95/P99
fallback rate
successful-task cost
~~~

路由器优化不能只看 FLOPs。若路由器本身需要运行一个大模型或触发频繁同步，节省的 attention 成本可能被决策开销抵消。

## 65.26 失败模式与修复路径

global 路径太少时，增加关键层或外部 retrieval；local window 太小时，扩大窗口或加入跨窗口 summary；state 过度遗忘时，调整 gate、保留显式近期记忆或增加回读；路径融合尺度不一致时，检查 norm、residual 和 gate 初始化；只有线上失败时，优先复现 padding、cache、preemption 和路由 trace。

高风险任务不应因为平均质量达标就关闭回退。若关键证据未能通过 citation support，系统应打开显式路径、要求用户提供范围、拒答或转人工。

## 65.27 hybrid 的层级调度比名称更关键

一个混合模型可能把局部 attention、全局 attention、SSM 或递归层按固定周期交替，也可能根据 token、任务或硬件动态选择。固定周期便于 kernel 和容量规划，动态路由可能提高灵活性，却增加路由开销、负载不均和可复现难度。阅读模型卡时应区分公开的层级比例与未披露的运行时策略。

设第 `l` 层的历史存储为 `m_l`、计算为 `f_l`，总资源可写成：

```math
M_{\mathrm{history}}=\sum_l m_l,
\qquad
F_{\mathrm{decode}}=\sum_l f_l.
```

局部层减少 `m_l`，递归层把历史压成 state，全局层承担跨段回读；但全局层仍可能决定长文检索能力。只看平均层数无法判断质量和服务成本。

## 65.28 hybrid 的路由反事实

构造四个对照：全 attention、全递归、固定混合、同资源动态混合。任务分为局部语言建模、远距复制、跨段问答、工具状态恢复和高并发 decode。若固定混合在远距任务保留质量、在 decode 上节省显存，说明层级分工有效；若动态混合只提高平均分却使尾延迟失控，就不适合默认上线。

## 65.29 混合状态的生命周期

同一个请求可能同时拥有局部 KV、全局 KV、latent cache 和递归 state。preemption、snapshot、迁移和 speculative rollback 必须对这些状态使用同一个 accepted prefix 和版本边界。只回滚显式 KV 而保留递归 state，会产生文本上难以复现的错误。

## 65.30 hybrid 的训练信号和部署信号

混合架构的训练可能依赖长序列并行、chunk scan 和 layer schedule，部署却受单 token decode、状态迁移和 kernel 覆盖约束。一个训练吞吐提升的方案，如果无法在 decode 中高效更新递归 state 或局部 KV，不会自动带来服务收益。

## 65.31 局部窗口的边界效应

窗口大小决定局部层能看到多少历史，也决定跨窗口信息必须经过多少全局或递归层传递。测试时应把关键事实放在窗口边界两侧，比较事实在窗口内、跨一个窗口和跨多个窗口的准确率。只测随机长文平均值，很容易漏掉边界退化。

## 65.32 混合架构的选择表

低延迟连续生成可偏向递归和局部层；需要精确引用和复杂工具 schema 的任务应保留全局/显式路径；高并发系统要优先验证状态 allocator、page、迁移和 p99。选择不是架构名称的排名，而是 workload、硬件和可验证能力之间的匹配。

## 65.33 小结

Hybrid attention 的核心是资源分工：局部路径处理近邻，递归状态处理可压缩的历史，全局或外部检索保留精确远程通道。它不是一个凭名称即可判断优劣的模块，而是一套依赖 schedule、状态生命周期、路由和评测的系统设计。只有当长程质量、短任务回归、尾延迟、峰值显存和回退成本同时可接受时，混合路径才算真正成立。
