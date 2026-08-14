# 第 62 章 Gated DeltaNet：用门控和 delta rule 更新线性状态

## 62.1 它要替代哪部分 attention

full attention 的优势是 query 可以直接访问完整历史，缺点是长序列的计算、带宽和 KV 状态昂贵。线性 attention 将历史累积为可更新的状态，降低显式 `T\times T` 交互；但简单累积容易把旧信息永久混合。

Gated DeltaNet 代表一种折中：delta rule 允许状态修正，gate 控制记忆衰减。Qwen 等公开模型资料中出现的 hybrid 路线，使这种状态化处理成为工程讨论的对象。具体模型是否使用何种 DeltaNet 参数化，仍要以模型卡、技术报告和代码核对。

## 62.2 线性 attention 的状态

用特征映射 `\phi` 表示 key 和 query，一个教学化状态为：

```math
S_t=\sum_{i\le t}\phi(k_i)v_i^{\top},\qquad z_t=\sum_{i\le t}\phi(k_i)
```

输出可以写成：

```math
y_t=\frac{\phi(q_t)^{\top}S_t}{\phi(q_t)^{\top}z_t+\epsilon}
```

它把逐 query 的历史访问转为递归更新，但也带来信息瓶颈：不同历史可能映射成相似状态，精确 token 检索不再天然存在。

## 62.3 delta rule 为什么不是简单追加

如果当前 key 为 `k_t`，状态对它产生的 value 预测为 `k_t^{\top}S_{t-1}`，当前观察值为 `v_t`，二者差值可以用于修正状态：

```math
\Delta S_t=\eta_tu_t(v_t-k_t^{\top}S_{t-1})
```

于是：

```math
S_t=S_{t-1}+\Delta S_t
```

这只是理解 delta correction 的简化式。真实实现可能使用多头矩阵、归一化、门控 value、特殊 decay 和 fused scan。面试时应讲清思想，不能把教学式等式当成某个模型的完整代码。

## 62.4 gate 控制记忆寿命

一个抽象门控更新为：

```math
S_t=g_t\odot S_{t-1}+\Delta S_t
```

如果 `g_t` 较小，旧状态快速衰减；如果较大，历史保留更久。它可以依赖当前 token、channel、head 或层。输入相关 gate 让模型能在重复日志和关键异常之间采用不同的记忆策略。

门控过强会忘掉远处事实，过弱会让 state drift 和饱和；gate 的数值稳定性还会受到量化、混合精度和长序列累积影响。训练监控应包括 gate histogram、state norm、non-finite 和不同长度 bucket 的 loss。

## 62.5 为什么常与 full attention 混合

递归路径擅长便宜扫描长历史，full attention 擅长精确对齐。一个 hybrid schedule 可以让多数层走 state path，少数层保留 global attention，为远距离证据提供直接通道；也可以用 local attention 处理邻近语法。

设第 `l` 层的路径是 `p_l`：

```math
p_l\in\{\mathrm{local},\mathrm{global},\mathrm{recursive}\}
```

真正的系统成本取决于每种路径的层数、head、窗口、状态大小和 kernel，不能从“有 Gated DeltaNet”直接推导。

## 62.6 重复日志中的信息分工

假设一百万 token 的日志大多是重复心跳，只有几处异常包含精确时间戳和错误码。递归层可以低成本扫描重复模式，global 层负责把异常字段与问题对齐。若所有层都递归，异常的具体时间戳可能被压缩；若所有层都 full attention，质量可能更稳但显存和 prefill 成本上升。

评测要同时问“是否找到异常”和“是否引用了正确时间戳”。只检查分类标签会掩盖递归压缩对精确归因的损失。

## 62.7 训练中的并行与稳定

递归更新有时间依赖，训练实现可能使用 chunk scan 或可结合的状态运算。chunk 边界必须与推理时的 state continuation 一致；否则模型训练时学会的是每段独立处理，线上却要求跨 chunk 记忆。

状态初始化、梯度截断、gate bias 和长样本配比都会影响稳定。短样本 loss 下降不代表长流状态稳定，应该监控 state norm、梯度 norm、长短 loss 差距和恢复一致性。

## 62.8 kernel 和硬件 trade-off

线性复杂度不保证 GPU 快。递归 update 可能有跨时间依赖，无法像大矩阵乘那样充分利用 GEMM；高效实现需要 scan、fused update、合理的 state layout 和不同 batch 长度的 padding 处理。

因此压测要分开看理论 FLOPs、memory traffic、kernel occupancy、tokens/s 和 p99。若 kernel fallback 到通用实现，理论优势可能消失；若状态更新占用大量带宽，增加 GPU 数也不一定能线性扩展。

## 62.9 与 Mamba/SSM 的边界

Gated DeltaNet、Mamba 和一般 SSM 都使用状态化序列处理，但更新方程、输入选择、卷积/扫描和 attention 交互不同。它们不能都被称为“线性 attention”。比较时应问：是否有显式 query-key 对齐，状态如何更新，能否直接读取历史，kernel 如何并行，以及是否和 full attention 混合。

## 62.10 评估设计

做四组任务：局部复制、远距离复制、冲突版本选择和长序列流式分类。改变长度、噪声、gate 初始化和 state precision，记录准确率、state norm、gate histogram、tokens/s、峰值显存和恢复一致性。

还要测 scheduler：请求完成后 state 是否释放，batch 重排是否保持结果，preemption 是否能恢复，speculative reject 是否回滚。架构质量和系统正确性必须同时过线。

## 62.11 常见失败

把 linearly bounded memory 误写成 linearly bounded 全部成本；gate 全部饱和；delta correction 数值不稳定；padding 更新 state；kernel 没有融合；local/global mask 边界 off-by-one；量化后 gate 和长程 recall 下跌；只测平均吞吐不测精确引用。

## 62.12 面试回答与练习

回答“Gated DeltaNet 解决什么问题”时，应说它用可修正的递归状态代替部分显式历史，gate 决定记忆保留与遗忘；相比 full attention，它降低长历史访问和状态成本，但牺牲部分任意检索能力并增加 state/kernel 复杂度。Hybrid schedule、长程任务和端到端压测决定是否值得。

练习一：实现一个 scalar gated state，观察 gate 变化如何影响远距离信息。

练习二：设计 local/global/recursive 三种路径的消融实验。

练习三：列出理论复杂度下降但实际 GPU 变慢的三种原因。

### 62.12.1 delta correction 在做什么

线性注意力或递归记忆通常维护一个聚合状态，更新便宜但难以精确删除旧错误。DeltaNet 类思路可以用当前输入产生的 correction 修正状态，使新观察不只是简单累加。一个抽象写法是：

```math
S_t=\Lambda_t(S_{t-1})+\Delta_t(x_t,S_{t-1})
```

门控版本再用 `g_t` 控制保留和更新：

```math
S_t=g_t\odot\Lambda_t(S_{t-1})
 +(1-g_t)\odot\Delta_t(x_t,S_{t-1})
```

具体张量形状、归一化和 gate 位置必须以论文或代码为准；公式的教学作用是说明“记忆”和“纠错”是两条不同的路径。

### 62.12.2 与 full attention 的能力边界

full attention 可以在当前 query 到来时重新访问每个历史 token；递归 state 只能访问已经压缩的摘要。面对局部连续信号、长流预测和大量重复上下文，递归路径可能很有效；面对多个远距离、相似实体和精确引用，压缩误差更容易暴露。

评估不能只测 perplexity。应增加键值回忆、版本消歧、数字复制、长距离反事实和引用支持任务，并比较 state 维度、序列长度和干扰强度。若 perplexity 变化很小但关键数字 recall 下降，说明平均语言损失没有捕获高价值信息损失。

### 62.12.3 hybrid schedule 的资源计算

假设有 `L` 层，其中 `L_g` 层使用显式 global attention，其余使用递归路径；粗略的显式 KV 相对比例可写为：

```math
\rho_{\mathrm{KV}}\approx\frac{L_g}{L}
```

这只是 payload 直觉。global 层可能有更多 heads，state 还需要 workspace 和通信；但它提醒我们，即使只保留少数 global 层，长上下文 KV 仍可能是主要成本。

### 62.12.4 kernel 现实

递归计算的理论复杂度较低，不代表 GPU 一定更快。小矩阵更新、非规则 gate、跨层 state 读写和 kernel launch 可能让 GPU 利用率低；full attention 反而有成熟的高吞吐 kernel。必须用真实 batch、长度和并发测端到端，而不是从 `O(T)` 和 `O(T^2)` 直接推断速度。

## 62.13 delta rule 的数值边界

delta 更新的直觉是用误差修正已有状态，但误差更新也可能放大数值噪声。若状态矩阵的范数持续增大，后续 query 的归一化可能失效；若 gate 和 decay 同时很小，模型又可能快速丢掉长程信息。

工程上要监控 state norm、update norm、gate saturation、非有限值和不同长度 bucket 的输出差异。混合精度下，先分别测 state 用高精度、低精度和分块 scale 的版本，再判断瓶颈来自算法还是数值格式。

门控初始化也有实际影响。若初始 gate 让所有 token 都强烈保留旧状态，模型可能很难学会写入；若初始 gate 让状态快速覆盖，长距离任务的梯度信号可能在早期消失。初始化、bias、梯度裁剪和长样本比例应作为同一组实验变量。

## 62.14 chunk scan 与并行化

递归更新看似只能串行，但训练时常把序列切成 chunk，在 chunk 内使用融合 kernel，在 chunk 之间传递边界 state。若更新满足可组合条件，可以用 scan 或并行 prefix 运算；如果 gate、归一化和 correction 依赖复杂的历史，则并行化会受限。

chunk 长度是一个资源折中。chunk 太短，边界 state 传递和 kernel launch 增加；chunk 太长，临时 workspace 和反向显存增加。比较时要同时记录训练吞吐、显存、长程 loss 和 chunk continuation 一致性。

推理端也要保持同一语义。一个常见错误是训练时在 chunk 内把 padding 排除，线上 fused kernel 却把 padding 作为有效输入更新 state。用不同 batch 长度和随机 padding 做 golden replay，通常能发现这种问题。

## 62.15 与线性 attention、SSM 的判别

看到“线性复杂度”时，先检查它保存的是 key-value 聚合矩阵、状态空间变量，还是压缩后的 attention 记忆。线性 attention 通常保留显式 query-key 特征映射，DeltaNet 增加状态纠正，SSM 更强调状态转移和输入选择；三者可以组合，但不是同义词。

一个实用的判别表是：当前 query 能否直接定位历史 token，状态更新是否由 key-value 误差驱动，是否存在显式卷积或状态转移矩阵，训练是否用 scan，推理是否需要 token 级 KV。回答这些问题，比背模型宣传词更可靠。

## 62.16 从公式走向工程验证

Gated DeltaNet 的知识核心是“delta correction + gated memory + serving state”。从公式到工程，至少要验证三件事：状态更新是否与离线全序列一致，gate 是否真的改变了记忆保留/擦除，而不是只响应 token 长度，以及状态大小和更新 kernel 是否在目标硬件上兑现了理论优势。

最小测试可以同时运行 full sequence 和 token-by-token 两条路径。给定同一组 `(k_t,v_t,q_t)`，逐步检查每个时间点的 `S_t` 和输出 `y_t`，并在中间位置插入 reset、padding、chunk boundary 和 batch reorder。若两条路径只在长序列或随机 padding 下分叉，问题通常在 state owner、有效长度或 fused kernel 的 mask，而不是 delta rule 本身。

还要做反事实任务：重复一个远处 token、在两个实体之间交替写入、插入无关干扰、显式要求遗忘旧版本。记录 exact recall、冲突选择率、状态范数、更新耗时、每请求 state bytes 和 p99。若 gate 关闭后质量不变，说明存在冗余路径；若状态很小但远程引用大幅下降，应把它定位为压缩/遗忘 trade-off。具体 gate 数值、层 schedule 和 kernel 仍须以对应官方实现为准，教学实验不能冒充模型内部事实。

## 62.17 Delta rule 的逐步推导

把线性注意力写成 fast-weight 记忆，可以得到一个很清楚的基线。令状态 S 保存 key 到 value 的线性映射：

~~~math
S_t=S_{t-1}+v_t k_t^{\mathsf T},
\qquad
\hat v_t=S_{t-1}k_t.
~~~

如果当前 key 在旧状态中已经对应了正确 value，就没有必要重复写入；如果预测值错误，则用误差修正：

~~~math
e_t=v_t-\hat v_t,
\qquad
S_t=S_{t-1}+\beta_t e_t k_t^{\mathsf T}.
~~~

这个更新可以被理解为一次低秩的在线回归。它不是把整个历史重新算一遍，而是让状态逐渐拟合“哪些 key 应该对应哪些 value”。因此 DeltaNet 比简单的累加式 linear attention 多了一个能力：对新观察到的关联进行纠错。

小白可以把 S 想成一张会改错的速记表。第一次看到“订单号 -> 状态 A”时写入 A；后来看到同一个订单号已经变成状态 B，delta error 会推动表格修正。深入分析时要注意，若不同实体的 key 在表示空间中很接近，修正一个实体可能干扰另一个实体；状态的容量和 key 的可分离性决定了它能否稳定存储多条关联。

## 62.18 gate 应拆成遗忘、擦除和写入

“有 gate”不是一个足够精确的描述。至少要区分三种作用：

1. 遗忘 gate：旧 state 保留多少。
2. delta gate：当前误差是否写入。
3. 输出 gate：递归路径的读出占最终输出多少。

一个教学级更新可以写成：

~~~math
\tilde S_t = D_t\odot S_{t-1}
              +B_t\odot(e_t k_t^{\mathsf T}),
\qquad
y_t=G_t\odot \operatorname{read}(q_t,\tilde S_t).
~~~

这里 D_t、B_t 和 G_t 不一定在真实模型中独立存在。这个分解的价值是帮助定位问题：如果旧事实过快消失，先查 D_t；如果新事实无法覆盖旧关联，查 B_t 和 key 表示；如果状态读出被另一条路径淹没，查 G_t 或残差尺度。

使用同一个标量 gate 同时控制擦除和写入实现简单，但表达力有限。2026 年公开的 Gated DeltaNet-2 预印本把“擦除”和“写入”解耦作为研究问题，摘要中明确讨论 channel-wise erase gate 与 write gate，并报告其在长上下文检索等任务上的实验信号。它是新的研究证据，不等于已经成为所有生产模型的默认方案；阅读时应把它作为路线演进和可验证假设，而不是泛化成产业共识。

## 62.19 与 KDA 的关系和差别

KDA、Gated DeltaNet 和其他线性注意力路线共享若干思想：固定或受控的递归状态、内容相关更新、遗忘机制以及 chunkwise scan。但名称相似不意味着张量公式相同。比较时建议列出以下字段：

| 维度 | 要核对的问题 |
| --- | --- |
| 读出 | state 是如何接受 query 的 |
| 修正 | 是否使用 value - prediction |
| 遗忘 | scalar、channel-wise 还是 block-wise |
| 写入 | 是否与擦除独立 |
| 状态 | 矩阵、向量、多个 head 还是混合 |
| 并行 | recurrent、chunkwise 还是双路径 |
| serving | 是否有显式 token KV、state snapshot 或两者 |

KDA 更常被描述为带通道衰减的递归 attention 路线；Gated DeltaNet 的教学重点是 delta correction 与 gate 的组合。二者可以在同一个比较框架下讨论，但没有足够资料时，不能从一个模型的名称推断另一个模型的内部实现。

## 62.20 状态更新的数值稳定性

递归 state 会经历比单个 token 更长的数值累积。即使每一步的 gate 都在 0 到 1 之间，反复的矩阵乘法和误差写入也可能使 state norm 爆炸或逐渐退化。常见稳定化手段包括：

1. 对 key、value 或 state 做 RMS normalization。
2. 在 readout 中使用归一化分母，避免 key 范数改变读出尺度。
3. 用 log-domain 表示衰减，减少长乘积下溢。
4. 对写入量设置数值保护，而不是简单裁剪所有状态。
5. 在 BF16/FP8 kernel 中保留更高精度的累积路径。

诊断不能只看 loss。训练时记录 state norm 的长度分布、异常 batch、gate 的极端分位数、梯度 norm 和 chunk boundary 的误差；推理时记录每个 request 的 state checksum、恢复前后的 norm 和输出差异。一个模型在短序列 loss 正常，不代表长序列 state 不会逐步发散。

## 62.21 chunkwise scan 的正确性

递归模型常见的工程落差是：论文中的递推公式正确，实际 GPU kernel 的 chunk 实现却在边界处少更新一次或多更新一次。最小测试应固定随机种子，建立三个实现：

~~~text
reference: token-by-token float64 recurrence
training: chunkwise implementation
serving: fused recurrent kernel
~~~

对相同输入比较每个 token 的 state、输出和梯度。不能只比较最后一个 token，因为中间位置的误差可能在后续被 gate 掩盖。若差异集中在每个 chunk 的第一个位置，优先查边界 state 的初始化；若差异随长度线性增加，优先查累积精度或衰减广播。

训练和服务还要在不同 batch 形状下比较：单请求、长度不同的 padded batch、packed batch、抢占恢复和跨 GPU 迁移。状态化模型的 batch 变化不应改变逻辑序列的结果。

## 62.22 一个状态更新的数量例子

设一维简化状态为 s=0.8，当前 key 对应的旧预测为 0.6，新 value 为 1.0，误差是 0.4。如果写入系数 beta=0.5、衰减 d=0.9、key 为 1，则：

~~~math
s_{\mathrm{new}}
=0.9\times0.8+0.5\times0.4
=0.92.
~~~

如果同一个 token 在没有 delta correction 的累加模型中直接写 value，状态可能变成 0.9 x 0.8 + 0.5 x 1.0 = 1.22，新旧关联的尺度变化更大。这个例子只是帮助理解“先预测、再按误差修正”的方向，真实状态是矩阵或多头张量，不能用一个标量例子推断实际精度。

更有意义的实验是让同一 key 依次绑定三个 value，观察模型是保留最新值、平均值还是出现振荡；再让相似但不同的 key 交替出现，测量实体间的串扰。这直接对应记忆容量和 key 可分离性，而不是只对应平均语言建模 loss。

## 62.23 硬件效率来自规律的内存访问

线性时间的复杂度并不自动等于更快。实际吞吐还取决于：

~~~math
\text{time}
\approx
\text{FLOPs}/\text{compute throughput}
+\text{bytes moved}/\text{memory bandwidth}
+\text{kernel launch overhead}.
~~~

递归 state 规模固定，但每个 token 仍可能需要读写状态；如果 state 很大、访问不连续或每步启动独立 kernel，理论上的线性复杂度可能被带宽和 launch overhead 吃掉。chunkwise scan 通过把多个 token 组织成块，减少 kernel 次数并提高矩阵乘法利用率，但会增加临时 buffer 和边界管理。

部署比较时要同时报告 prefill 和 decode。对长 prompt，chunked prefill 的并行效率可能更重要；对单 token decode，多 request 共享 kernel 和状态布局更重要。只报告总 tokens/s 会掩盖短请求首 token 延迟和长请求尾延迟。

## 62.24 Gated DeltaNet-2 作为研究演进案例

截至 2026-08-06，arXiv 查询可以看到题为 Gated DeltaNet-2 的公开预印本。其摘要提出把遗忘和写入从共享标量中分开，并讨论带通道衰减的快速权重更新和 chunkwise WY 算法。这个结果很适合用来说明研究路线的演进：

1. 先用线性状态降低历史访问成本。
2. 用 delta rule 处理新旧关联冲突。
3. 用 gate 控制遗忘和状态更新。
4. 继续拆分 gate 的职责，降低擦除与写入耦合。
5. 用硬件友好的 chunk 算法把数学更新变成可训练 kernel。

但预印本摘要中的 benchmark 结果仍受模型大小、数据、硬件、实现和评测 harness 约束。读者不能把单一报告的长上下文收益写成“Gated DeltaNet 已全面胜过 Transformer”；正确结论是：它提供了一个可验证的结构改进假设，需要在同等训练和 serving 条件下比较。

## 62.25 递归状态与显式 KV 的组合

生产模型可能同时保留递归 state 和少量显式 KV。state 负责压缩历史，local KV 负责近邻语法，global retrieval 或 sparse attention 负责远程精确证据。此时缓存账本应写成：

~~~math
M_{\mathrm{total}}
=N_{\mathrm{layer}}
\left(M_{\mathrm{state}}
+M_{\mathrm{localKV}}
+M_{\mathrm{metadata}}\right).
~~~

状态路径减少的不是所有内存，而是 token 维度的某一部分。若 local window 很大，KV 仍可能主导；若 state head 很多，state 也可能成为显存瓶颈。调参应按请求长度和并发分别测，而不能用单一平均比例外推线上容量。

## 62.26 评测：记忆、遗忘和回读要分开

一个完整评测至少包含三类曲线：

| 曲线 | 关注点 |
| --- | --- |
| 写入曲线 | 新事实进入 state 后能否被读取 |
| 遗忘曲线 | 无关干扰增加后旧事实保留多久 |
| 回读曲线 | 能否返回原始位置、版本和引用 |

写入成功而回读失败，说明状态能支持语义预测但不支持审计；遗忘成功但新事实覆盖失败，说明 gate 过于保守；短序列成功、长序列失败，可能是 state 容量、数值误差或训练长度问题。把这三类失败混成一个 accuracy，会使工程修复方向完全错误。

## 62.27 delta rule 的直觉：先纠错，再保留

线性注意力把历史压进一个累计矩阵，容易把新信息简单叠加。DeltaNet 类更新可以理解成先根据当前 key 预测旧状态对 value 的响应，再用误差修正状态。教学化地写成：

```math
\hat v_t=S_{t-1}k_t,
\qquad
e_t=v_t-\hat v_t,
\qquad
S_t=S_{t-1}+\eta_t e_t k_t^{\mathsf T}.
```

`eta_t` 控制更新强度。它不是“免费记忆”，而是在有限状态中选择哪些关联值得被改写。若 key 不可区分、value 噪声很大或 gate 长期饱和，更新就会变成遗忘或污染。

## 62.28 门控的三个失败区间

门控接近零时，模型几乎不吸收新证据，长任务会出现 stale state；门控接近一且没有衰减时，旧模式可能持续覆盖新模式；门控在 batch 内剧烈振荡时，状态更新带宽和数值稳定性会成为瓶颈。评测应记录 gate 的均值、方差、饱和比例以及它们与任务错误的相关性。

一个实用的诊断是把 gate 固定为常数、关闭 delta 修正、只保留标准线性路径做消融。若关闭 gate 后质量反而更稳，问题可能在训练稳定性或状态尺度，而不是架构概念本身。

## 62.29 Gated DeltaNet 的训练到 serving 契约

训练时可以用并行 scan 或 chunk 化算法，推理时却必须维护正确的递归边界。chunk 切分、padding、packed sequence、reset、speculative rollback 都必须给出同一状态语义。若训练 kernel 的边界处理与 serving kernel 不同，loss 可能正常而长对话结果错位。

发布前用单 token 逐步更新、chunk 更新和恢复更新三条路径比较 state checksum、logits 和最终输出。只比较最后一段文本会掩盖状态在中途已经漂移的问题。

## 62.30 小结

Gated DeltaNet 的核心不是“把 attention 变成一个线性公式”，而是把历史记忆拆成可修正、可遗忘、可读出的状态更新。它的优势依赖三个前提：key 表示能区分重要关联，gate 不会错误遗忘，递归公式能被稳定地并行训练和高效执行。任何一个前提失败，线性复杂度都不能自动转化为更好的模型或更低的线上成本。

模型卡、技术报告和代码公开到什么程度，决定本文能写到哪里。Gated DeltaNet-2 等后续研究可以作为结构演进的证据，但不应替代对具体模型 revision、硬件和评测条件的核验。
