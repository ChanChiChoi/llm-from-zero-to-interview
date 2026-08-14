# 第 61 章 KDA：把历史表示成可门控的递归状态

## 61.1 从“保存每个 token”开始

标准 decoder attention 在生成第 `t` 个 token 时，需要访问历史的 K/V。它几乎不丢信息：query 可以直接对某个远处 token 分配权重，引用和调试都比较容易解释。代价是历史状态随序列长度增长，长上下文和高并发会把显存、带宽和调度压力推到很高。

KDA（常被解释为 Kimi Delta Attention 或相关的门控 delta attention 路线）代表另一种思考：不把全部历史 token 留在显式 cache 中，而是按顺序把历史更新成一个有限大小的状态。这个状态仍然是缓存，只是缓存的单位从 token 变成了递归 state。

这里要先画出边界。公开资料可以支持某个模型采用 KDA 或类似架构方向；具体 state 维度、gate 位置、delta 更新和层间 schedule，如果没有完整技术报告或代码，不能从缩写推断。

## 61.2 从线性 attention 到递归更新

一种教学化的线性 attention 会维护矩阵状态：

```math
S_t=\sum_{i\le t}\phi(k_i)v_i^{\top}
```

查询时使用：

```math
y_t=\frac{\phi(q_t)^{\top}S_t}{\phi(q_t)^{\top}z_t+\epsilon},\qquad z_t=\sum_{i\le t}\phi(k_i)
```

这样就不需要显式构造每个 query 与全部 token 的 score 矩阵。可是简单累积会把旧信息不断叠加，无法主动删除错误或过时内容。Delta rule 的直觉是，当前 token 不只是“再写一条”，还可以根据当前 key 对旧状态做修正：

```math
S_t=S_{t-1}+\eta_tu_t(v_t-k_t^{\top}S_{t-1})
```

式中的差值表示“当前状态对这个 key 的预测和新 value 有多大偏差”。真实 KDA 的矩阵形状和门控细节可能不同，上式只用来说明递归更新和纠错的关系。

## 61.3 gate 解决什么问题

如果所有历史都永久累积，状态会越来越混杂。门控让模型决定旧状态保留多少、当前信息写入多少，以及某些通道是否应该重置。一个抽象形式是：

```math
S_t=g_t\odot S_{t-1}+(1-g_t)\odot\widetilde S_t
```

`g_t` 可以按 token、channel、head 或 layer 计算。它接近 1 时更像保留旧记忆，接近 0 时更像用新状态覆盖。实际参数化还可能把 gate 放在 key、value、delta 或输出路径，而不是简单地套在整个矩阵上。

门控有两个相反的风险。门控过强，远处证据在无关 token 经过后被遗忘；门控过弱，状态饱和，错误信息持续污染后续请求。训练时观察 gate 的均值和饱和比例，往往比只看最终 loss 更早发现问题。

## 61.4 一个远距离比较例子

文档开头介绍实体 A 的三个属性，中间插入五万 token 的无关日志，末尾介绍实体 B，并要求比较 A 和 B。如果使用显式 KV，最后的 query 可以直接找到 A 的相关 token。若使用递归 state，模型必须在中间处理日志时保留 A 的关键属性，或者借助某个全局/显式层重新读取。

这不是简单的“递归一定失败”。如果训练任务经常要求压缩后复述关键属性，state 可以学会保留；但它的成功率会随干扰数量、属性类型和 gate 行为变化。实验要同时测精确数字、实体名、顺序和冲突事实，不能只测主题摘要。

## 61.5 显式 KV 与 KDA 的取舍

显式 KV 的状态大小近似为：

```math
M_{\mathrm{KV}}\approx2BLTH_{\mathrm{kv}}d_hb
```

KDA 的状态更接近：

```math
M_{\mathrm{state}}\approx BLd_sb_s
```

其中 `d_s` 是递归状态的有效元素数，通常不随 `T` 线性增长。后者可能显著降低长流的内存，但不代表总成本必然更低：递归更新有跨时间依赖，kernel 可能不如大矩阵 GEMM 高效，state snapshot 也可能增加调度复杂度。

需要精确引用、任意跳转和多证据归因的任务更依赖显式路径；流式分类、重复日志扫描和固定状态控制可能更适合递归路径。混合架构往往让两条路径分担不同工作。

## 61.6 训练计算图的变化

full attention 可以在训练时并行计算整个序列；递归 state 天然按时间更新，训练实现要使用 scan、chunked recurrence 或可并行化的 associative update。state 的初始值、chunk 边界和反向传播截断都会影响结果。

如果训练和推理的 state reset 规则不一致，模型在长流服务中可能出现训练时没有的漂移。packed batch 还要确保 padding 不更新 state，否则一个样本的空位会改变它后续的记忆。

## 61.7 Serving 的 state contract

对一个请求，状态接口至少应包含：

```text
value: tensor
step: integer
model_revision: string
dtype: string
checksum: string
```

prefill 结束后产生 state，decode 每步更新；完成或取消时释放，preemption 时 snapshot/restore，speculative decode 时使用临时副本。状态不能只挂在 batch slot 上，因为请求会被重排、合并和拆分。

## 61.8 一个状态一致性实验

用同一 prefix 做四次运行：单请求、两个 batch slot、暂停后恢复、speculative 接受/拒绝。比较每层 state checksum 和最终 logits。若单请求与 batch 结果不同，先查 padding mask、state index 和完成请求是否仍在更新；若恢复不同，查 snapshot 是否包含所有层、step、dtype 和位置契约。

再把输入切成前缀 `x_{1:a}` 和后缀 `x_{a+1:T}`，比较一次运行与“先运行前缀、保存 state、再继续”的输出。二者不一致时，问题通常是 state serialization 或 reset，而不应先归因于模型“记忆不足”。

## 61.9 评估信息压缩而非只评估速度

长程 recall、顺序交换、冲突版本、精确数字、引用位置和工具参数是核心质量指标。系统指标包括 state size、snapshot size、state update latency、batch throughput、p99、恢复时间和跨 worker 迁移。

还要测租户隔离：完成请求后复用 batch slot，输入一个与前一请求相关的查询，确认旧 state 不会泄露答案。这类测试既是架构测试，也是安全测试。

## 61.10 常见错误

把“固定状态”写成“没有 cache”；认为递归 state 可以任意定位原文；只删除 K/V 张量却没有修改 scheduler；padding 更新 state；preemption 只保存 token offset；拒绝 draft 后不回滚；跨模型版本复用旧 state；把社区对 KDA 的猜测当成官方结构。

## 61.11 面试回答与练习

面试官问“KDA 解决什么问题”，可以回答：它用带门控和 delta 更新的递归状态压缩历史，目标是降低长上下文的显式 KV 成本；代价是历史信息经过压缩，任意 token 读取和归因能力可能下降，且 serving 需要管理 state 的 reset、snapshot、rollback 和版本。最终要用长程任务和端到端 SLO 验证。

练习一：实现一个带遗忘 gate 的 toy state，比较不同衰减对远距离 recall 的影响。

练习二：设计 batch 重排和 preemption 的 state checksum 测试。

练习三：列出三种任务，分别适合显式 KV、递归 state 和 hybrid 路径，并说明理由。

### 61.11.1 递归状态的基本计算

把显式历史 `H_t` 压缩成固定大小 state `S_t`，可以抽象为：

```math
S_t=F(S_{t-1},x_t;g_t),\qquad
y_t=G(S_t,x_t)
```

`g_t` 是控制保留、遗忘或更新强度的门。与显式 attention 保存每个 token 的 K/V 不同，递归路径的状态大小可以近似不随序列长度增长；代价是过去的信息被混合，未来查询不能任意恢复所有原始 token。

一种更具体的教学形式是：

```math
S_t=g_t\odot S_{t-1}+(1-g_t)\odot U(x_t,S_{t-1})
```

这不是 KDA 的完整实现，只用来说明 gate 如何控制旧状态和新更新的比例。实际架构可能使用矩阵状态、delta correction、归一化和不同的投影路径。

### 61.11.2 精确访问和压缩记忆的分工

如果问题要求从一百万 token 中逐字引用一个数字，显式 KV 或外部检索更适合；如果任务是持续处理日志流、维护最新摘要或在固定状态上预测下一步，递归 state 可能更省资源。KDA 类方向的价值不是让所有任务都用 state，而是让系统可以按任务选择路径。

### 61.11.3 worked example：遗忘 gate 的影响

假设标量 state 的更新为 `S_t=gS_{t-1}+(1-g)x_t`。当 `g=0.99` 时，旧信息保留时间较长，但新事件进入状态较慢；当 `g=0.5` 时，新事件很快覆盖旧状态，近期预测可能变好，远程引用可能变差。真实模型的 gate 是按维度、token 和层变化的，不能用单个标量代表完整行为，但这个例子说明 gate 是能力和成本之间的参数，而不是可解释性标签。

### 61.11.4 state 的事务要求

推测解码、batch 重排和 preemption 都可能产生暂时 token。state 必须区分 `committed_state` 和 `temporary_state`；候选被拒绝时回滚临时更新，request 取消时释放 owner，恢复时校验 model revision、step、dtype 和 checksum。否则一个请求的递归历史可能污染另一个请求。

## 61.12 递归状态的容量与可恢复性

递归 state 的固定大小并不意味着它能保存任意多的信息。可以把 state 看作一个有限带宽的写入介质：每个新 token 都要决定哪些维度继续保留、哪些维度被覆盖，以及旧信息是否还能被当前 query 区分。

设 state 有 d_s 个有效维度，输入历史有 T 个 token。显式 KV 的可访问条目随 T 增长，而递归 state 的可访问表示数近似固定。若任务需要同时保留 n 个互相独立的实体，n 增长到超过 state 能够分离的范围时，碰撞概率会上升。这个结论不是一个简单的维度下界，因为模型可以利用结构、压缩和任务先验；它只是提醒我们不要把固定内存误写成无限记忆。

可恢复性也有两层。第一层是从 state 继续生成，要求数值和位置连续；第二层是从 state 反推出原始证据，通常不成立。一个 state 能让模型继续预测下一个 token，不代表系统能够回答“这条结论来自原文第几行”。

## 61.13 训练长度与任务分布

递归架构是否擅长长流，取决于训练中是否反复出现“信息写入—长时间干扰—信息读取”的任务结构。只把短句串接成长序列，会增加序列长度，却不一定训练出有用的状态管理。

训练集应包含至少四种样本：近期信息覆盖旧信息、旧信息需要长期保留、多个实体交替出现，以及错误信息需要被新版本纠正。每种样本都要标记 gold state event 或 gold evidence，便于区分忘记、覆盖和读取失败。

chunk 训练还要比较两种模式：每个 chunk 都 reset state，以及前一个 chunk 的 state 传给下一个 chunk。前者适合独立样本，后者才接近线上长流。若只使用第一种，模型可能在单 chunk benchmark 上表现良好，却在持续服务中产生状态漂移。

## 61.14 一个可诊断的长流实验

可以把一个长流分成四个阶段，并在阶段之间插入可控干扰：

~~~text
write key-value fact
  -> repeat distractors
  -> introduce conflicting version
  -> query exact value and source
~~~

每次实验保存 state norm、gate histogram、最后一次看到证据的位置、答案、引用和 state checksum。改变干扰长度、重复比例、state precision 和 reset 位置，画出成功率曲线。

如果主题答案正确但版本判断错误，说明 state 保留了语义但没有保留冲突细节；如果答案正确但引用失败，说明递归路径可能需要外部 provenance；如果单请求正确而 batch 失败，则优先查 runtime state owner 和 padding。

## 61.15 从递归公式走向系统实现

KDA 的核心不是一个缩写，而是“用有限、可门控的递归状态代替部分显式历史”的架构选择。它可能改善长流的状态成本，却把信息保真、state lifecycle 和 kernel 问题带进系统。Kimi 等模型的公开资料可支持公开的架构方向；具体 gate、delta、层表和实现应以官方技术报告或源码为准。

## 61.16 从线性注意力到 delta rule

理解 KDA 最稳妥的起点不是模型名称，而是一个可读的 fast-weight 记忆。设每个 token 产生 key、value 和 query，先用外积把 key-value 对写入矩阵状态：

~~~math
S_t = S_{t-1} + k_t v_t^{\mathsf T}
~~~

读取时用当前 query 查询状态：

~~~math
r_t = S_t^{\mathsf T} q_t
~~~

这条路径的优点是历史被合并到固定大小的矩阵中，推理时不需要保存每个历史 token 的 K/V。缺点也很明显：两个相似 key 写入不同 value 时，状态会叠加，旧信息可能被污染；当 query 需要精确找回一个旧 token 时，固定状态没有显式地址可以跳转。

Delta rule 的思路是先读取旧状态对当前 key 的预测，再只写入预测误差。令：

~~~math
\hat v_t = S_{t-1}^{\mathsf T} k_t,
\qquad
e_t = v_t-\hat v_t
~~~

那么更新可以写成：

~~~math
S_t = S_{t-1} + \beta_t k_t e_t^{\mathsf T}
~~~

如果状态已经能正确预测当前 value，误差接近零，写入量也接近零；如果当前 token 与旧记忆冲突，delta 更新会修正关联。这里的公式是理解 delta memory 的教学抽象，不等于某个闭源模型的完整实现。实际实现还可能有归一化、分头状态、衰减、数值稳定项和不同的张量转置约定。

## 61.17 门控衰减究竟改变了什么

仅有 delta correction 仍然可能让旧信息长期留在状态里。KDA 类路线引入门控衰减，把“保留旧记忆”和“接受新写入”分开控制。一个直观的通道级表达是：

~~~math
S_t = D_t \odot S_{t-1} + \beta_t k_t e_t^{\mathsf T},
~~~

其中 D_t 是由当前输入产生的衰减张量或其低秩参数化，0 <= D_t <= 1，\odot 表示逐元素或按通道作用。若 D_t 接近 1，历史保留得更多；若接近 0，历史快速遗忘。实际模型可能使用 log-domain、sigmoid、指数参数化或硬件友好的变形，不能把这个简式当作公开模型的逐行代码。

小白可以把它想成一块有很多格子的白板。普通线性注意力不断往白板上贴纸条，delta rule 先擦掉与新事实冲突的旧关联，gate 再决定哪些格子应该褪色。深入分析时需要追问：gate 是按 token、head、channel 还是 state block 生成的？衰减和写入是否共用一个标量？更新是否保持可并行的 chunk 形式？这些问题决定了它到底是一个可部署的 kernel，还是只有数学上的递归表达。

门控还有一个容易被忽略的副作用。遗忘不是免费的：如果一个关键事实在很长的无关文本后仍然需要被引用，过强的衰减会把它变成不可恢复的残影；如果衰减太弱，错误的用户指令会持续影响后续决策。因此 gate 的均值、方差、饱和比例和与任务失败的相关性，都应作为训练和推理诊断信号。

## 61.18 KDA 与标准 softmax attention 的信息地址差异

标准 causal attention 在第 t 步可以访问历史 token 的显式 K/V：

~~~math
y_t = \sum_{j\le t}
\operatorname{softmax}_j
\left(\frac{q_t k_j^{\mathsf T}}{\sqrt{d_h}}\right)v_j
~~~

它的内存随历史长度增长，但每个历史位置仍有相对清晰的地址。递归状态路径则把历史压缩成 S_t：

~~~math
y_t = g_t \odot \operatorname{read}(q_t,S_t)
      +(1-g_t)\odot \operatorname{local}(q_t,K_{\le t},V_{\le t}).
~~~

后一个式子表示常见的混合设计空间，不表示 KDA 必然采用该精确融合。两条路径的差异可以用三个问题理解：

1. 能否直接跳转到任意历史位置？
2. 多个相似事实发生冲突时，状态能否保留版本和来源？
3. 历史变长时，内存增长发生在 token 维度，还是固定状态维度？

如果任务是扫描长日志并判断是否出现某种模式，固定状态可能很划算；如果任务要求从一百万 token 中找出带行号的原始配置并比较三次修改，显式检索、局部 attention 或外部索引仍然重要。递归状态降低的是访问成本，不是自动生成可审计的原文地址。

## 61.19 一个可计算的状态容量例子

假设某个递归 head 的状态是 d_k x d_v = 64 x 64，使用 BF16，每个元素 2 字节，则单 head 状态约为：

~~~math
64\times64\times2=8192\ \text{bytes}=8\ \text{KiB}.
~~~

即使处理一百万 token，这部分状态仍不会像完整 KV 一样按 token 线性增长。但这不代表它“免费记住”了一百万 token。状态只有 4096 个数，必须把许多历史模式投影到同一组自由度里。若任务要求保留 200 个实体、每个实体 10 个字段和 4 个版本条件，真正的记忆需求可能远大于一个小状态能稳定表达的容量。

工程容量模型至少要把三项分开：

~~~math
M_{\mathrm{request}}
=M_{\mathrm{state}}
+M_{\mathrm{local\ KV}}
+M_{\mathrm{workspace}}.
~~~

M_state 是递归状态，M_local_KV 是保留给局部或全局 attention 的显式缓存，M_workspace 是 kernel 临时 buffer、量化 scale、batch metadata 和通信缓冲区。只计算第一项，会低估实际显存；只看到显存下降，又可能忽略 state precision 降低导致的质量损失。

## 61.20 训练时的递归与并行

递归定义天然适合逐 token 推理，却不天然适合 GPU 训练。若完全按照：

~~~text
for t in range(sequence_length):
    state = update(state, token[t])
    output[t] = read(state, query[t])
~~~

执行，训练会被 Python 级时间循环限制。现代线性注意力和 DeltaNet 类方法通常寻找 chunkwise 或 scan 形式：在一个 chunk 内整理矩阵运算，在 chunk 之间传递边界 state。这样既保留递归语义，又让 GPU 批量计算更多 token。

训练实现需要验证三件事：

1. chunk size 改变时，输出是否与逐 token reference 在容许误差内一致。
2. backward 是否正确穿过 state boundary，而不是只在 chunk 内传播梯度。
3. 混合精度和长序列下，state norm 是否出现爆炸、消失或异常饱和。

可以用一个小规模 reference 做回归。设序列长度为 17，分别用 chunk size 1、4、8、17 运行同一组随机输入，比较每个位置的输出和参数梯度。若只有 chunk size 1 正确，说明并行化公式或 boundary state 有 bug；若前向一致而梯度不一致，问题通常在自定义 backward 或保存的中间状态。

## 61.21 状态生命周期是 serving 协议的一部分

递归架构把“模型状态”从隐含的 KV cache 中分离出来，服务端必须明确它属于谁。至少需要记录：

| 字段 | 作用 |
| --- | --- |
| request id | 防止不同请求复用状态 |
| conversation revision | 多轮历史发生变化时识别旧状态 |
| logical position | 记录状态已经处理到哪个位置 |
| model revision | 权重或 state layout 改变时禁止复用 |
| dtype and quantization | 解释数值误差和兼容性 |
| reset boundary | 新样本、工具调用或权限域的状态边界 |
| checksum | 检测传输、快照和恢复是否损坏 |

一次请求被 preempt 后，恢复的不只是“最后生成的文本”，还包括 state、logical position 和随机采样状态。一个很危险的错误是只保存文本前缀，重新把文本喂入另一个版本的模型或不同的 state path。表面上服务可以继续输出，实际上概率分布已经不再等价。

多租户场景还要考虑状态隔离。用户 A 的隐私信息如果写入共享 state，即使后续输出没有直接复制原文，也可能通过行为偏置泄露。状态缓存的 key 不能只用 prompt hash，还要包含租户、权限域、模型版本和策略版本。

## 61.22 KDA 路径的混合架构取舍

纯递归路径的优势是状态规模稳定、长流吞吐好、decode 访问规律；纯显式 attention 的优势是任意历史检索和表达能力直接。混合架构把两者分工：

| 任务特征 | 更需要的路径 | 原因 |
| --- | --- | --- |
| 连续日志分类 | 递归 state | 模式扫描和历史摘要占主导 |
| 短距离语法 | local attention | 近邻对齐成本低 |
| 精确远程引用 | global/full attention 或检索 | 需要显式地址 |
| 长周期对话偏好 | state + 外部 memory | 状态提供流畅性，外部存储提供可追溯性 |
| 工具参数校验 | 显式 attention + schema verifier | 一个字段错误可能产生副作用 |

层 schedule 也不是越稀疏越好。若 global layer 太少，远程信息无法进入输出；若过多，递归路径节省的显存和带宽被重新消耗。应按任务做 ablation：固定参数量和训练 token，改变递归层比例、global layer 周期和 local window，报告长程检索、短任务、吞吐和 p99。

## 61.23 从状态读出并不等于可解释

递归状态可以支持下一个 token 预测，却不天然提供人类可读的记忆条目。要回答“模型为什么认为版本 3 是最新的”，至少需要一种额外路径：

1. 保存写入 state 的来源范围，允许回读原文。
2. 建立 state-to-evidence 探针，测状态变化和证据遮挡的关系。
3. 使用外部检索或结构化 memory 保存原子事实。
4. 在高风险任务上要求模型输出引用，并由 verifier 检查。

一个简单的因果实验是：先输入包含唯一事实的文档，再输入不同长度的无关干扰，最后询问事实。比较完整路径、删除事实路径和强制清空 state 的输出。如果删除事实后答案仍然正确，可能是模型记忆了训练分布，也可能是干扰没有真正覆盖关键路径；如果答案正确但引用不存在，说明语义预测与证据追溯已经分离。

## 61.24 最小 reference pseudo-code

~~~python
state = zeros(d_key, d_value)
for token in sequence:
    key, query, value = project(token)
    predicted = state.T @ key
    error = value - predicted
    decay = sigmoid(decay_project(token))
    state = decay[:, None] * state + beta * outer(key, error)
    readout = state.T @ query
~~~

阅读这段代码时要注意三点。第一，decay 的广播方式决定它是按行、按列还是按元素衰减。第二，真实 kernel 会把多个 token 合并成块矩阵，不能直接把 Python 循环当作性能实现。第三，state 的初始化和 reset 是协议的一部分；若新样本沿用了旧 state，任何架构比较都会失真。

## 61.25 一套可复现的 KDA 评测

评测应至少包含四种任务，而不是只测 perplexity：

1. 顺序任务：交换两个事件，要求输出先后关系。
2. 远程检索：唯一 needle 分别放在开头、中间、结尾和多个干扰区。
3. 冲突版本：旧版本、新版本和例外条件分开出现，要求输出版本、条件和引用。
4. 流式恢复：在随机位置保存 state，恢复后继续生成，并与不间断运行比较。

每个长度和位置桶都记录：

~~~text
answer accuracy
evidence recall
citation support
state norm
gate saturation
TTFT
TPOT
peak memory
snapshot restore error
~~~

最终报告不能只写“长上下文更快”。如果 state memory 降低了，但 citation support 在中间位置下降，部署结论应是“适合摘要或扫描，不适合无回读的审计”；如果吞吐提升来自更激进的量化，则要把量化误差与架构收益分开做对照。

## 61.26 KDA 的信息瓶颈如何被测量

递归状态的优势来自不保存全部历史，代价是历史必须通过有限维状态传递。可以构造三类合成任务：只需记住最近 token 的任务、需要累计统计的任务、需要从很远位置精确复制字符串的任务。前两类可以检验状态更新和门控，第三类用来观察压缩表示的上限。

若状态维度为 `d_s`、序列长度为 `T`，显式 KV 的历史存储随 `T` 增长，而递归 state 的主项近似与 `d_s` 和层数相关；但计算不一定更便宜，因为每步仍要读写 state，且状态更新可能受内存带宽限制。报告时应同时给出 state bytes/token、update time、长距离复制准确率和 batch 扩展曲线。

## 61.27 KDA 与标准 attention 的反事实对照

公平对照不能只比较相同参数量。应固定 tokenizer、训练数据和输出预算，分别比较 full attention、KDA、KDA 加少量全局层以及 KDA 加外部检索。若 KDA 在连续流预测上延迟更低，却在随机引用任务上下降，说明差异来自历史表示能力；若加全局层后恢复，说明混合路径正在补偿 state 的压缩损失。

训练和推理也要分开测：训练可能受并行扫描、序列长度和 kernel 影响，decode 则受 state update、batch 和调度影响。不能从论文中的理论复杂度直接推导线上 TTFT 或 TPOT。

## 61.28 本章结语

KDA 类递归注意力的真正价值，是重新安排“历史如何被保存、遗忘和读取”的资源账本。它把按 token 保存的显式历史，换成固定或缓慢增长的 state，再用 delta correction 和 gate 处理新旧信息冲突。代价是信息地址变得间接，状态生命周期、可追溯性、kernel 和评测都必须由系统补齐。

截至 2026-08-06 可核验的公开资料，KDA、Kimi 系列具体配置和层间组合仍应以对应版本的模型卡、技术报告和实现为准。本文的 fast-weight 公式用于建立可迁移的理解框架，不将未公开的内部细节写成确定事实。
