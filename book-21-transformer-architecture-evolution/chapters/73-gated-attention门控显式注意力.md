# 第 73 章 Gated Attention：让显式 attention 路径学会“什么时候介入”

## 73.1 为什么显式 attention 需要门控

混合模型中，递归或局部路径已经处理了大部分上下文，显式 attention 可能只在精确检索、跨段连接或特殊 token 上有价值。如果显式路径始终以固定权重加入，既可能浪费计算，也可能干扰状态路径的稳定表示。

Gated Attention 用一个可学习的 gate 调节显式路径的贡献。它的含义不是“模型明确知道这个 token 重要”，而是当前参数化更依赖哪条信息通路；要得到解释，需要把 gate 与证据位置、attention weight 和任务结果一起分析。

## 73.2 基本形式

设局部或递归输出为 `y_s`，显式 attention 输出为 `y_a`：

```math
y=g\odot y_a+(1-g)\odot y_s,\qquad g=\sigma(Wx+b)
```

gate 可以按 token、channel、head、layer 或 query 计算。也可以让 gate 调节 value、residual 或 attention logits，而不是只融合两个输出。不同放置位置会改变梯度和计算成本。

## 73.3 gate 的三个作用

第一是信息选择：精确远程证据可能需要提高显式路径权重。第二是稳定：在大多数局部任务上保留 state path，避免 global attention 引入噪声。第三是容量分配：模型把更多显式计算留给难样本。

三者不能混为一谈。gate 较大只表示输出融合偏向显式 attention，不能证明它找到了金标准证据；gate 变化也可能由数值尺度或 layer norm 造成。

## 73.4 一个长日志例子

重复心跳附近，局部路径已经足够；当问题要求把末尾异常与开头发布版本对齐时，显式 attention 可能提供直接通道。理想情况下 global path 的 gate 在这类 query 上升，其他位置保持较低。

实验可比较固定 `g=0`、固定 `g=1` 和 learned gate，记录异常召回、时间戳引用、TPOT、显存和 gate histogram。如果 learned gate 的质量接近 `g=1` 但速度接近 `g=0`，才说明动态路径有价值。

## 73.5 训练中的饱和与正则

若 `g` 全部接近 1，模型可能退化成 full attention；若全接近 0，显式路径没有被使用。可以检查初始化、gate bias、梯度流、不同任务的 gate 分布和 layer norm 尺度。

正则化可以鼓励 gate 不过早饱和，但不应为了让直方图好看而强行分散。最终仍要用任务质量和成本判断 gate 是否学到了有用的分工。

## 73.6 serving 的影响

如果 gate 只改变融合权重，计算图仍可能要执行两条路径；如果 gate 能跳过显式 attention，就涉及动态分支和 kernel efficiency。动态分支可能让 GPU 利用率下降、p99 抖动，或破坏 batch 内的统一执行。

量化也会改变 gate 的数值范围。speculative decode、local/global cache 和 state snapshot 要保证两条路径的临时状态都能 commit/rollback。

## 73.7 可解释性边界

分析 gate 时至少保存 layer、head、token position、任务类型、证据标签和 attention weight。比较 gate 高的 token 是否真的与证据重合，并做遮挡实验：删除证据后 gate 是否改变，输出是否受影响。

即使相关性很高，也不能直接说 gate 是因果证据选择器。它可能只是对输入长度、标点或 token 类型敏感。

## 73.8 评估与失败模式

按局部语法、远程实体、长日志、冲突证据和工具参数分桶。记录答案、引用、gate entropy、路径 FLOPs、TTFT、TPOT、p99 和显存。

失败包括 gate 全一/全零、量化后饱和、路径 shape 错、动态分支导致 kernel fallback、state 与显式 KV 不同步、把 gate heatmap 当作解释证明。

## 73.9 面试回答与练习

回答“Gated Attention 是什么”时，应说它用 gate 控制显式 attention 与局部/递归路径的组合，目标是在精确检索和低成本状态之间分配计算；需要同时评估 gate 分布、任务质量、kernel、显存和延迟，不能只看最终准确率。

练习一：实现两个 toy attention path 和一个 gate，画出不同任务的 gate histogram。

练习二：设计一个能区分“gate 相关性”和“证据因果性”的遮挡实验。

练习三：解释为什么 gate 动态跳过路径可能降低平均 FLOPs却提高 p99。

### 73.9.1 gate 如何进入 attention

门控可以作用于 attention output、value、logit 或 residual。以输出门控为例：

```math
o_t=g_t\odot o_t^{\mathrm{attn}}
 +(1-g_t)\odot o_t^{\mathrm{other}}
```

若 gate 由 query、层状态或任务条件决定，它可能在不同输入上选择不同路径。要理解实现，必须确认 gate 的 shape、归一化、训练目标和推理时是否真正跳过计算；仅仅把 output 乘小，不一定节省 kernel FLOPs。

### 73.9.2 gate 与可解释性

gate 大只能说明某条参数路径贡献被放大，不能说明模型“认为某个证据重要”。要做解释实验，需要同时遮挡证据、改变 gate、比较 attention、输出和任务结果。若 gate 改变但答案不变，可能存在冗余路径；若答案改变而 gate 不变，信息可能通过其他通路进入。

### 73.9.3 路径稀疏的系统代价

动态 gate 如果造成大量小分支，会引入 kernel launch、分支不均衡和 batch 分裂。平均 FLOPs 下降，p99 可能因为少数复杂请求触发全路径而上升。服务端要记录 gate histogram、各路径 token 数、kernel time、GPU 利用率和尾延迟，而不是只统计理论乘加量。

### 73.9.4 训练和回归

gate 训练可能出现塌缩：所有 token 都选同一路径，或者 gate 高频切换导致不稳定。可加入负载均衡、稀疏正则或路径预算，但这些目标不能损害关键长程任务。评估应做全路径、固定 gate、随机 gate 和按任务 gate 的消融。

### 73.9.5 gate 的简单参数化

一个教学模型可以令：

```math
g_t=\sigma(W_g h_t+b_g)
```

其中 `\sigma` 把 gate 限制在 `0` 到 `1`。这只说明 gate 可以由当前表示产生，不能推断具体模型采用 sigmoid、softmax、向量 gate 还是离散路由。训练时还要决定 gate 是否参与梯度、是否有温度、是否有稀疏或负载约束。

### 73.9.6 如何判断 gate 真的节省计算

比较四条路径：始终执行显式 attention、始终执行另一条路径、计算后再乘 gate、根据 gate 真正跳过 kernel。只有第四条才可能减少主要 FLOPs；第二和第三条只是改变输出混合。压测时要固定 batch、长度和 GPU，记录 kernel 时间、显存读写、路径比例和 p99。

### 73.9.7 长上下文和高风险任务

如果 gate 在大多数普通 token 上关闭 global path，却在关键证据 token 上打开，平均成本可能下降；但关键 token 的 gate 错误会造成不可接受的引用失败。评估要把 gate 干预与原子证据遮挡结合，给高风险任务设置最低 evidence recall 和 citation support，而不是只按平均 gate 稀疏率优化。

## 73.10 gate 的位置决定它控制什么

gate 作用在 logits 上，会改变注意力分布；作用在 value 上，会改变被读取的信息；作用在 output 或 residual 上，只改变路径融合。三种 gate 的计算成本和梯度行为不同。

如果 gate 在 logits 上，必须检查 mask、温度和数值范围；如果 gate 在 value 上，可能仍然要完整计算 attention；如果 gate 只在 output 上，通常不会自动省掉显式 kernel。论文中的“门控 attention”必须结合公式和代码判断。

## 73.11 gate 与事实证据的因果实验

要判断 gate 是否真的参与证据选择，可以做四组干预：保留证据并打开 gate，删除证据并保持 gate，保留证据但强制关闭 gate，删除证据并关闭 gate。比较答案、引用、attention、gate 和路径成本。

若删除证据后 gate 仍然高，gate 可能响应的是长度或 token 类型；若关闭 gate 后答案仍正确，其他路径可能有冗余；若只有打开 gate 且保留证据时成功，才支持它在该任务中的功能性作用。

## 73.12 动态 gate 的部署策略

连续 gate 适合融合但未必节省 FLOPs；离散 gate 更可能跳过 kernel，却会带来 batch 分裂。可以按 request、chunk 或 token block 做路由，减少细粒度分支；代价是一些不需要 global path 的 token 仍会一起执行。

上线条件应同时设置平均成本、p99、关键任务 recall、gate collapse 和 fallback 比例。为了追求平均稀疏率而损害少数高风险 token，通常是不合理的优化。

## 73.13 gate 的尺度和残差稳定性

两条路径的输出尺度不同，会让 gate 看起来像在做路径选择，实际却只是在补偿数值范围。比较 gate 之前，应记录两条路径的 norm、layer norm 后的分布和 residual 的大小。

可以做一个尺度消融：固定 gate，分别把显式和递归输出归一化到相同 RMS，再观察任务质量和 gate 学习结果。如果 gate 分布显著改变，说明原来的 gate 可能被表示尺度驱动。

## 73.14 训练、推理和量化回归

训练时 gate 可以使用连续值，推理时为了节省 kernel 可能离散化或按 block 决定路径。离散化阈值会改变关键 token 的路径，量化又会改变 gate logits。应比较连续 full path、连续 gate、离散 gate 和量化离散 gate 的长程 recall、引用和 p99。

高风险任务的 gate 不能只按平均稀疏率优化。关键证据 token 如果走错路径，应触发显式回读或人工复核。

## 73.15 gate 的位置决定它控制什么

若 gate 乘在 attention logits 上，它可能改变竞争关系；若 gate 乘在 value 或 output 上，它更像路径输出过滤；若 gate 与 residual 融合，它控制的是 attention 分支对主干的写入。三种位置不能用同一个“注意力门控”公式代替。

实现和模型卡若没有明确 gate 位置，正文只能讨论设计空间。实验中应记录 gate 前后的 logits、value norm、residual norm 和梯度，避免把尺度变化误判成内容选择。

## 73.16 gate 的稀疏性和质量风险

连续 gate 即使数值很小，也可能仍然执行完整 attention，只有离散或 block gate 才可能节省 FLOPs；离散化又会导致 batch 分裂和临界 token 误路由。稀疏率必须和关键任务 recall、p99、fallback 一起报告。

对于数字、代码、权限和工具参数，不能只优化平均 gate 成本。关键证据若被 gate 关闭，应触发显式回读或 verifier，而不是让模型用语言流畅度掩盖信息缺失。

## 73.17 gate 的训练和部署一致性

训练中使用连续 gate、推理中使用阈值 gate 时，模型看到的路径分布不同。量化还可能改变 gate logits，使边界 token 在部署中换路。应对连续、离散、量化和 fallback 四种模式做一致性回归。

状态层面，gate 决定了哪些 KV/state 被更新。speculative decode 回滚时必须恢复 gate 统计、cache 和递归 state；只回滚输出文本会留下不可见的路径差异。

## 73.18 从 gate 定义走向执行路径

Gated Attention 是路径组合机制，不是自动的可解释性机制。一个 gate 只有在反事实实验中改变了可解释的读取行为，并且端到端成本确实下降时，才有工程意义。应把 gate 所在位置、连续/离散形式、是否减少实际 kernel 工作、是否改变 KV/state 写入和回滚语义分别记录。

最小验证可以包含四组：保留关键证据并打开 gate、删除关键证据并打开 gate、保留证据并关闭 gate、删除证据并关闭 gate。再把数字、代码标识符、版本冲突和普通主题各做一组。如果 gate 只依赖长度或 token 类型，而不区分证据是否支持答案，它更可能是容量/尺度控制器；如果 gate 影响答案但不减少任何 kernel 计算，它是质量路径而不是 serving 稀疏优化。

部署时还要做连续 gate、阈值 gate、量化 gate 和 fallback 的一致性回归。speculative decode 拒绝候选时，输出、gate 统计、显式 KV 和递归 state 必须一起回滚。具体 gate 在 logits、value、residual 还是 output 上，以模型实现为准；未公开的结构不能从一段输出反推。

## 73.19 gate 的三种数学位置

门控显式 attention 常见的教学形式有三类：

1. logits gate：改变 attention score，再做 softmax。
2. value gate：改变读取到的 value。
3. residual gate：在 attention 输出与其他路径之间做融合。

logits gate 可以写成：

~~~math
s_{t,j}
=\frac{q_t k_j^{\mathsf T}}{\sqrt{d_h}}
+b_{t,j},
\qquad
\alpha_{t,j}=\operatorname{softmax}_j(s_{t,j}).
~~~

value gate 则可以写成：

~~~math
y_t=\sum_j\alpha_{t,j}(g_{t,j}\odot v_j).
~~~

residual gate 形式是：

~~~math
y_t=g_t\odot y_t^{\mathrm{attn}}
+(1-g_t)\odot y_t^{\mathrm{other}}.
~~~

这三者都可能被口头称作 Gated Attention，但梯度、归一化和 serving 成本不同。没有实现证据时，不能把一个形式替代另一个。

## 73.20 gate 不是注意力权重

attention weight 表示某个 query 对历史 key 的归一化匹配；gate 表示某条路径或某个表示分量的缩放。一个 token 的 attention weight 很高，gate 可能仍然关闭；gate 很高，也不代表某一个证据 token 被选中。

解释实验要同时记录：

~~~text
attention weight
gate value
value norm
residual norm
output change
task result
~~~

只有在遮挡证据、干预 gate 和比较输出后，才能讨论 gate 是否参与了任务因果。单纯画 gate heatmap 容易把长度、标点和 hidden scale 当成“重要性”。

## 73.21 gate 的初始化和退化

如果 gate 初始接近 0，模型一开始几乎只使用另一条路径，可能导致 gated branch 梯度弱；如果 gate 初始接近 1，另一条路径可能无法学习。训练后还可能出现三种退化：

1. 全开：模型没有获得计算节省。
2. 全关：新增路径没有贡献。
3. 按长度开关：gate 只学到 prompt length shortcut。

可以统计 gate 的熵：

~~~math
H(g)
=-g\log g-(1-g)\log(1-g).
~~~

熵低不一定坏，关键要看它是否与任务难度、证据位置和质量相关。若所有任务都全开或全关，应检查 gate loss、初始化、归一化和路径输出尺度。

## 73.22 门控与计算预算

如果 gate 只在输出融合时缩放，attention 计算已经发生，gate 不会节省 FLOPs；如果 gate 在执行前决定是否打开某条路径，才可能节省计算，但会带来动态 shape、分支和 batch split。应区分：

~~~text
representation gate: changes values after compute
execution gate: changes which kernels run
cache gate: changes which history is stored
~~~

三者的收益和风险不同。执行 gate 如果误判远程证据，质量可能突然崩溃；cache gate 如果提前淘汰信息，后续无法恢复；表示 gate 更安全但节省的计算有限。

## 73.23 一个反事实验证

假设模型在长文档中找到正确版本。做四组运行：

1. 保留证据，原始 gate。
2. 删除证据，原始 gate。
3. 保留证据，强制关闭显式路径。
4. 删除证据，强制打开显式路径。

记录答案、引用、attention、gate 和延迟。若删除证据后答案不变，可能是先验或数据泄漏；若关闭路径后答案不变，可能有 state 或 RAG 冗余；只有保留证据且打开路径时质量稳定，才说明该路径在该任务上有功能贡献。

## 73.24 gate 与混合架构

在 hybrid architecture 中，gate 可能负责在 local、global、recursive 和 latent 路径之间分配信息。若每条路径输出尺度不同，应先做 norm 或 learned projection，再融合；否则 gate 可能只是在补偿数值尺度。

一个路径融合账本可以写成：

~~~math
y
=\sum_{r\in\mathcal R}g_r \cdot P_r(y_r),
\qquad
\sum_{r\in\mathcal R}g_r=1.
~~~

这里 R 是路径集合，P_r 是对齐维度和尺度的投影。路径权重和为 1 是一种教学约束，实际实现可能不满足。评估时要记录每条路径的 activation rate、FLOPs、显存、证据召回和引用支持。

## 73.25 门控和量化

gate logits 往往对尺度敏感。权重量化、KV 量化和激活量化可能改变 logits 排序，使原本在边界上的 token 换路。量化回归应比较：

~~~text
full precision gate distribution
quantized gate distribution
path activation difference
answer difference
citation difference
P99 difference
~~~

如果只看最终文本，少数关键参数错误可能被采样掩盖。对代码、工具参数和数字任务，应使用 greedy 或固定 seed，比较 token-level logits 和结构化输出。

## 73.26 gate 和安全策略

在安全系统中，不能让模型通过 gate 自己关闭安全检查。高风险动作的 policy、权限和人工确认必须是硬约束；gate 只能决定模型内部信息路径或是否请求额外检查。若安全分类器低置信度，应增加检查或降级，而不是让 gate 放宽。

trace 中应保存 gate revision、路径决策、policy decision、工具权限和最终动作。这样才能区分“模型没有看到证据”和“策略拒绝了动作”。

## 73.27 训练与 serving 的一致性

训练中使用连续 gate、推理中使用阈值 gate 时，模型看到的路径分布不同。训练中每条路径都有梯度，推理中却可能只执行一条 kernel，导致 quality gap。应做四种一致性回归：

1. continuous gate。
2. threshold gate。
3. quantized gate。
4. fallback gate。

推测解码回滚时还要恢复 gate 统计、cache 和递归 state；只回滚输出文本会留下不可见的路径差异。

## 73.28 gate 的收益要和位置/归一化分开

显式 attention 上增加 gate 可能改善信息写入、抑制噪声或控制残差幅度，但同一版本若同时修改 norm、初始化和训练长度，就无法知道收益来自哪里。最小消融应包含无 gate、固定 gate、可学习 gate、不同初始化和不同位置长度。

## 73.29 gate 饱和时的 serving 影响

gate 接近零或一不一定是错误，可能表示模型正在选择保留或放行；但大面积饱和会降低动态范围，也可能让量化和低精度更敏感。线上记录 gate 分布、激活范数、异常 token、p99 和质量切片，不能只看平均 logits。

## 73.30 门控路径的回退和发布

若新 kernel、dtype 或 gate artifact 不兼容，可以回退到无 gate 或旧 checkpoint，但要重新验证 cache、位置和输出协议。回退前后应使用同一 golden prefix 比较事件和任务质量，并明确哪些模型 revision 支持门控。

## 73.31 小结

Gated Attention 的难点不是写出一个 sigmoid，而是证明 gate 改变了有价值的计算路径，且没有破坏证据、位置、缓存和安全边界。分析时要区分 attention weight、表示 gate 和执行 gate；部署时要把路径成本、量化、回滚、fallback 和 P99 纳入同一套验证。
