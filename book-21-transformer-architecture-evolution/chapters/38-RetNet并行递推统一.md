# 第三十八章：RetNet：Retention 机制与并行/递推统一

## 38.1 RetNet 想统一什么

RetNet（Retentive Network）提出一种 retention 机制，希望同一模型既能用并行方式训练，也能用递归方式推理，还可以使用 chunkwise 形式在两者之间折中。它借鉴 attention 的多头表示，又把历史交互写成带衰减的状态。

核心问题仍是：训练需要吞吐和并行，在线 decode 需要固定状态和低延迟。若三条路径数学/数值不一致，模型会在离线 benchmark 与线上流式服务之间产生行为差异。

## 38.2 Retention 的基本公式

一个简化 retention 可以写成：

~~~math
Y=(QK^\top\odot D)V
~~~

其中 `D` 是按相对距离衰减的因果矩阵：

~~~math
D_{ij}=\gamma^{i-j}\mathbb{1}[j\le i],\qquad 0<\gamma<1
~~~

它可以展开为递归状态：

~~~math
S_t=\gamma S_{t-1}+K_t^\top V_t
~~~

~~~math
y_t=Q_t S_t
~~~

多头 retention 使用不同的 `\gamma_h` 或分组，让不同 head 负责不同时间尺度。

## 38.3 小白直觉：带年龄折扣的注意力账本

普通 attention 在每次提问时重新比较所有历史；retention 先把每条历史写入账本，越旧的记录折扣越大。不同 head 有不同折扣：有的关注最近对话，有的保存较长期趋势。

它比只保留固定窗口更平滑，但比完整历史更容易丢失精确细节。衰减不是“记忆长度”的唯一参数，状态维度和训练也很重要。

## 38.4 并行、递推、chunkwise 三条路径

并行路径直接计算带衰减的下三角关系，适合训练；递推路径逐步更新 `S_t`，适合在线；chunkwise 路径在 chunk 内并行，并把 chunk 摘要传给下一个 chunk。

设 chunk c 的输入/输出状态关系为：

~~~math
S_{\mathrm{out}}^{(c)}
=A_c S_{\mathrm{in}}^{(c)}+B_c
~~~

只要正确保留 `S_out`，chunk 边界不会重置历史。错误的状态初始化会让 chunkwise 模式退化为独立窗口。

## 38.5 Multi-Scale Retention

多尺度的直觉是让不同 head 使用不同衰减：

~~~math
\gamma_h\in\{\gamma_1,\gamma_2,\ldots,\gamma_H\}
~~~

小 `\gamma` 强调近期，接近 1 的 `\gamma` 保留更长趋势。若所有 head 的衰减相同，模型的时间尺度多样性下降；若某些 head 接近 1，状态数值和旧信息污染风险上升。

## 38.6 一个 toy retention

~~~python
import torch


def retention(q, k, v, decay):
    # q/k: [batch, seq, dim], v: [batch, seq, value_dim]
    state = torch.zeros(q.size(0), q.size(-1), v.size(-1))
    outputs = []
    for step in range(q.size(1)):
        state = decay * state + k[:, step].unsqueeze(-1) * v[:, step].unsqueeze(-2)
        outputs.append(torch.bmm(q[:, step:step + 1], state))
    return torch.cat(outputs, dim=1)


q = k = torch.randn(2, 5, 4)
v = torch.randn(2, 5, 3)
print(retention(q, k, v, decay=0.95).shape)
~~~

这个实现不含归一化、head、relative position 和高效 kernel，只用于验证递推 state 的 shape。

## 38.7 与 attention 的能力比较

retention 的状态包含衰减后的 key-value 累积，不能保证对任意历史位置保留独立 token 表示。它对趋势、重复模式和流式状态有优势；对多个相似实体、精确数字、冲突版本和后置 query 重解释要专项测。

full attention 可以根据当前 query 改变历史排序；retention 的历史影响部分通过衰减和聚合提前确定。chunkwise 可以降低计算，却不能恢复已经被状态聚合的信息。

## 38.8 训练/推理和硬件

并行训练路径可能使用下三角矩阵、分块 kernel 或 prefix algorithm；递推推理只读写 state。GPU 对大矩阵友好，递推路径对小 batch 可能受 launch 和 memory bound 影响。真实收益要按 prompt、output、batch 和 head 数测。

状态 serving 需要处理 reset、snapshot、rollback、batch reorder 和 model revision。推测解码时，候选 token 更新的 state 必须能回滚，否则拒绝候选会污染后续输出。

## 38.9 失败模式

包括 decay 方向写反、mask 允许未来、并行/递推数值不一致、chunk 边界重置、不同 head 时间尺度未生效、padding 更新 state、state 量化溢出和 prefix sharing 错误。

测试先对比短序列三种路径的 logits，再增加长度、chunk、dtype 和 batch；记录 state norm、decay histogram、精确复制率和恢复误差。

## 38.10 机制与边界：retention 是结构化的 attention-like 计算

retention 保留了 query/key/value 的角色，但把历史交互加入固定衰减矩阵。它位于 full attention、linear attention 和递归 state 之间：有内容相关读写，却受聚合结构限制。多尺度 decay 是 inductive bias，不是任意注意力模式。

阅读论文时要关注并行/递推的等价条件、归一化、位置/decay 的参数化、训练稳定和 chunkwise 复杂度。不要把 retention 的公式直接当作所有后续模型的实现。

## 38.11 面试追问、误区与练习

**问：RetNet 如何做到训练并行、推理递归？**

标准回答：带衰减的 query-key-value 交互可以写成并行矩阵形式，也可以按递归 state 累积；chunkwise 在两者之间折中。工程上必须验证三条路径的数值和边界一致。

**问：Retention 与 linear attention 的关系是什么？**

标准回答：二者都可能维护聚合 state，但 retention 显式引入按距离的衰减和多尺度机制，具体归一化、状态和 attention-like 结构不同，不能直接视为同一个实现。

常见误区包括忽略 decay、把递推 state 当完整 KV、只测并行路径，以及没有测试拒绝候选后的 rollback。

练习：实现多 head、不同 decay 的 retention，比较并行抽象、递推和 chunkwise 的输出及长程证据 recall。

## 38.12 retention 的衰减和信息保真

Retention 的衰减因子决定历史贡献如何随距离下降。衰减太快，远程实体和版本难以保留；衰减太慢，状态混合和噪声增加。multi-scale retention 通过不同尺度分工缓解这个问题，但仍然存在容量边界。

实验要单独改变衰减尺度，测局部语法、远程复制、冲突选择、工具参数和 state size。不能只看一个平均 perplexity。

## 38.13 三种执行形式的一致性

RetNet 的并行、递推和 chunkwise 形式如果用于同一个 checkpoint，输出应在误差门限内一致。用同一 prefix 做三条路径的 logits 对照，可以定位 state 初始化、chunk 边界和归一化 bug。

## 38.14 chunkwise 的边界状态

chunkwise retention 不是简单地把长序列切段后分别运行。每段需要接收上一段的状态，并在边界保持衰减、归一化和位置连续。若状态传递被截断，长程任务会退化成多个独立短任务。

测试应改变 chunk 长度和切分位置，要求一次性计算与分块计算输出一致。再加入 preemption 和恢复，确认保存的不仅是 token offset，还包括每个 retention head 的 state。

## 38.15 retention 的服务资源

递推路径减少了按 token 保存的历史，但每个请求仍需 state、workspace、snapshot 和调度元数据。高并发下 state update bandwidth 可能成为瓶颈；低并发下 kernel launch 可能抵消理论复杂度收益。

## 38.16 retention 的归一化与长度偏差

简化公式 S_t=γS_{t-1}+K_t^T V_t 只描述了信息累积，没有说明输出是否需要除以一个同样累积的权重。若不同长度的序列使用相同 state 读取规则，长序列可能因为历史总量更大而改变激活尺度，而不是因为内容本身改变。

可以额外维护一个标量或向量归一化状态：

~~~math
z_t=\gamma z_{t-1}+K_t^\top \mathbf{1},\qquad
\tilde y_t=\frac{Q_tS_t}{Q_tz_t+\varepsilon}
~~~

实际模型可能使用不同的归一化、门控和位置参数化，不能把这个式子当作 RetNet 的完整实现。它的教学价值在于提醒读者：并行矩阵公式、递推状态和输出尺度必须一起比较。只验证方向一致而不验证数值尺度，会把长度偏差误判成模型能力。

## 38.17 多尺度 retention 的分工实验

多头衰减的意义可以通过控制实验看出来。固定主干参数，只改变各 head 的 γ_h 分布，分别测局部语法、远程复制、冲突实体和工具参数。若所有 head 的曲线几乎相同，说明多尺度没有真正形成分工；若慢衰减 head 的 state norm 过大，则需要检查旧信息污染和归一化。

一个实用的报告表至少包含：

| 实验 | 改变因素 | 质量指标 | 系统指标 |
|---|---|---|---|
| 单实体远距复制 | 距离、衰减 | exact match | state bytes |
| 多实体冲突 | 实体数、位置 | 版本选择率 | p99 |
| 流式恢复 | chunk、snapshot | logits 误差 | restore time |
| 批处理 | batch、padding | 输出一致性 | update bandwidth |

这样可以把“多尺度记忆”从架构名词转成可观察的行为。

## 38.18 拒绝候选时的状态回滚

Retention state 是历史的函数，候选 token 一旦写入，就会改变后续状态。设接受前状态为 S_t，草稿提出 r 个 token，候选更新为 S_hat；若主模型只接受前 a 个 token，正确状态应是：

~~~math
S_{t+a}=F(S_t,u_{t+1:t+a})
~~~

而不能继续使用 S_hat。实现可以保存完整 snapshot、保存每步增量，或使用可逆/分支状态；选择取决于 state 大小和回滚频率。测试时要专门构造“第一候选被拒绝、第二候选被接受”的场景，否则推测解码可能在普通样例上看似正常，却在拒绝率高时逐步漂移。

## 38.19 Retention 的三种计算路径

RetNet 类方法常从并行、递归和块递归三种视角理解：并行路径适合训练整段序列，递归路径适合逐 token 推理，chunked 路径在二者之间折中。三种路径只有在同一衰减、位置、归一化和初始状态契约下才应近似一致。

对每种路径记录输出误差、梯度、状态大小、吞吐和峰值显存。若并行训练和递归推理差异随长度增长，优先检查 decay、mask、chunk boundary、dtype 和状态初始化，而不是马上把问题归因于模型能力。

## 38.20 衰减状态的能力边界

衰减使旧信息逐渐降低权重，适合趋势和稳定上下文，但对精确版本、离散实体和任意远程引用可能不够。多尺度 decay 能覆盖不同记忆寿命，却仍是压缩表示；它不能自动保存每个 token 的可寻址副本。

混合设计可以保留少量 full/global attention 或外部检索，处理需要精确证据的 query。评测要区分摘要、局部语法、远程复制、reset 和多证据合并，不能只报告长序列 perplexity。

## 38.21 一个递推/并行对照

把简化 retention 写成：

```math
S_t=\gamma S_{t-1}+k_t v_t^{\top},
\qquad
y_t=q_tS_t.
```

递推路径逐 token 更新 `S_t`，并行路径可以把所有历史贡献写成带衰减的下三角矩阵。两者在理想精度下表达同一形式，但 serving 还要处理 chunk 边界、state snapshot、batch reorder 和不同长度请求。对照实验应比较 logits、state checksum、吞吐、p99 和取消恢复，而不是只比较训练 loss。

## 38.22 状态容量和衰减并不是同一件事

当 `\gamma` 接近 1 时，历史贡献衰减得慢，但状态中的多个实体也更容易混合。可以用半衰期描述单个时间尺度：

```math
j_{1/2}=\frac{\log 0.5}{\log\gamma}
```

它只说明一个衰减系数的时间尺度，不说明模型能精确读取多少事实。真正的能力还取决于 state 的维度、key 的可分离性、value 的表达、归一化和训练任务。多尺度 head 需要通过控制实验验证是否形成了分工，而不是从一组 `\gamma` 数字直接推断长期记忆。

## 38.23 chunk 边界的工程回归

把长序列切成 chunk 后，必须传递上一段的 state、位置和归一化信息。最小回归可以固定一个 prefix，在不同 chunk 长度和不同切分位置运行一次性路径与 chunkwise 路径，比较每个 token 的 logits、最终输出和 state checksum。

还要加入 preemption、取消和恢复。若只保存 token offset，恢复后重新计算的 state 可能与原路径不同；若只保存 state 而忽略 position 或 dtype，结果可能在短样本上接近、在长样本上漂移。chunkwise 不是简单的窗口拼接，而是一个状态协议。

## 38.24 retention 与完整 attention 的路由

Retention 适合把历史压成有时间尺度的摘要，但对精确版本、多个相似实体和任意远程引用有容量边界。完整 attention 可以按 query 重新访问历史，代价是随上下文增长的计算和 KV。工程上可以让 retention 处理连续流，让少数 global attention 层或外部检索处理高价值证据。

路由决策要按任务切片：摘要和趋势识别、远程复制、冲突实体、工具参数、引用审计分别测成功率、state bytes、TTFT/TPOT 和恢复一致性。不能只用 perplexity 证明 retention 已替代完整 attention。

## 38.25 证据范围与部署边界

RetNet 论文支持 retention 的并行、递推和 chunkwise 统一目标；具体多尺度参数化、归一化、kernel 和部署接口需要按实现核对。本文的 `S_t` 递推是帮助理解信息累积的教学抽象，不等于某个生产 checkpoint 的全部公式。

如果推测解码、分页 allocator 或量化进入 retention serving，候选 state 的提交、回滚和跨请求隔离都要单独验证。固定状态降低了历史存储，却增加了状态版本和恢复责任。

## 38.26 Retention 的路由与精确证据

Retention 用衰减和累积保存历史，适合趋势、局部模式和持续流；但当 query 需要在多个同名实体中重新选择某一个原子事实时，聚合状态可能不如显式 attention。可以让 retention 负责全局扫描，再在高风险或候选分歧时触发 retrieval/attention，并测触发率、额外延迟、引用支持和恢复成本。

推测解码或 beam search 还要处理候选 state 的提交和回滚。被拒绝的候选不能继续污染 retention 累积；chunk boundary、衰减归一化和 batch reorder 必须绑定 request owner。只有并行、递推和 chunkwise 三种路径都通过 golden replay，统一性才具有工程意义。

## 38.27 从并行矩阵形式推到递归状态

RetNet 最值得掌握的不是“有三种模式”这一结论，而是三种模式为什么能够来自同一个加权和。忽略 head、归一化和投影，令 `k_i` 是第 `i` 个位置的 key，`v_i` 是 value，则到位置 `t` 的状态可以展开为：

~~~math
S_t
=\sum_{i=1}^{t}\gamma^{t-i}k_i^{\top}v_i.
~~~

把最后一项拆出来，就得到递推式：

~~~math
S_t=\gamma S_{t-1}+k_t^{\top}v_t.
~~~

当前 query 读取状态：

~~~math
y_t=q_tS_t.
~~~

如果把所有 `q_t`、`k_i` 和衰减项一次性排列，就得到带因果下三角衰减矩阵的并行计算；如果逐 token 更新 `S_t`，就是 recurrent 路径。二者共享的是同一个贡献权重 `gamma^(t-i)`，而不是“训练版和推理版碰巧输出相似”。

这也解释了能力边界：`S_t` 是 key-value 的聚合矩阵，不是历史 token 的独立副本。只要两个实体被写入相同或相近的状态方向，后续 query 就可能无法把它们分开。并行化改变的是执行顺序，不会凭空恢复已经在状态中合并掉的信息。

## 38.28 一个可手算的 retention 例子

用标量 `k`、`v` 和 `q=1` 做一个最小例子。设 `gamma=0.5`，三个历史项为 `(k,v)=(1,2),(2,1),(1,3)`。未归一化状态的递推为：

~~~math
S_1=2,
\qquad
S_2=0.5\times2+2\times1=3,
\qquad
S_3=0.5\times3+1\times3=4.5.
~~~

如果还维护一个与 key 对齐的归一化量 `R_t`：

~~~math
R_t=\gamma R_{t-1}+k_t,
\qquad
\hat y_t=\frac{S_t}{R_t+\varepsilon},
~~~

则 `R_1=1`、`R_2=2.5`、`R_3=2.25`，对应的输出约为 `2.0`、`1.2`、`2.0`。第二个 token 虽然 value 是 `1`，但它的 key 较大，改变了聚合的权重；第三个 token 又通过衰减后的历史和当前 value 改写状态。

这个例子有两个用途。第一，它让“衰减账本”变成可检查的数值，而不是口号。第二，它说明归一化不能事后随便补上：并行路径、递推路径和 chunkwise 路径必须使用相同的分子、分母、初始化和 epsilon 语义。实际模型的多头、旋转位置和投影更复杂，但边界错误往往就发生在这个最小结构上。

## 38.29 chunkwise 为什么不是简单的窗口拼接

设一个 chunk 有 `m` 个 token，进入 chunk 前的状态为 `S_in`。把 chunk 内部展开，可以写成：

~~~math
S_{\mathrm{out}}
=\gamma^mS_{\mathrm{in}}
 +\sum_{j=1}^{m}\gamma^{m-j}k_j^{\top}v_j.
~~~

因此可以记成：

~~~math
S_{\mathrm{out}}=A_mS_{\mathrm{in}}+B_m,
\qquad A_m=\gamma^m.
~~~

在矩阵、多头或可学习衰减的实现里，`A_m` 可能是按 head/维度组织的线性变换，但含义不变：chunk 内部并行计算 `B_m`，chunk 之间递归传递边界状态。若每个 chunk 把 `S_in` 置零，系统得到的就不是 chunkwise retention，而是许多互不相干的短序列。

工程上至少要传递三类边界：数值 state、逻辑位置/衰减步数和 padding mask。只传 state 不传位置，会使位置相关参数在恢复后错位；只传位置不传 state，则相当于丢失历史。对变长 batch，还要确保已经结束的序列不会继续向别的序列更新状态。

## 38.30 归一化、长度偏差与状态污染

衰减 state 解决了 KV 随上下文增长的问题，却带来两个相反的风险。`gamma` 太小，远端证据快速消失；`gamma` 太大，旧实体持续留在状态中，新的 query 可能受到污染。半衰期：

~~~math
j_{1/2}=\frac{\log 0.5}{\log\gamma}
~~~

只描述权重下降到一半所需的步数，不等于模型能精确记忆 `j_{1/2}` 个事实。归一化还可能让不同长度的状态具有相近幅度，却掩盖内容碰撞；所以要同时看 state norm、实体冲突准确率、远距复制和无目标误报。

padding 是常见的污染来源。若 batch 中短序列的 padding 仍然执行 `S_t=gamma S_{t-1}+k_t^T v_t`，它会改变后续 state；若 mask 只遮住输出而没有遮住状态更新，最终 logits 可能只在长样本上发生漂移。正确测试需要把 padding、reset、batch reorder、抢占和恢复放在同一组 golden replay 中，而不是只测等长输入。

## 38.31 从“统一计算路径”到可部署的选择标准

RetNet 的统一性只有在三种路径同时满足四个条件时才具有工程价值：

1. **数值一致**：同一 prefix 下，parallel、recurrent 和 chunkwise 的 logits 误差低于任务验收条件。
2. **状态可恢复**：snapshot/restore、抢占迁移和 batch reorder 不改变后续输出。
3. **资源可预测**：状态字节数、kernel workspace、TTFT、TPOT 和 p99 随长度/并发的增长已测量。
4. **能力可接受**：长程复制、冲突版本、多证据合并和引用支持没有因聚合而越过业务阈值。

可以定义一个简单的回归记录：

~~~math
E_{\mathrm{path}}=\max_t\lVert \ell_t^{(\mathrm{parallel})}
-\ell_t^{(\mathrm{recurrent})}\rVert_\infty,
\qquad
E_{\mathrm{restore}}=\max_t\lVert \ell_t^{(\mathrm{full})}
-\ell_t^{(\mathrm{restored})}\rVert_\infty.
~~~

其中 `ell_t` 是 logits。仅比较最终生成文本会把小的数值差异隐藏在采样噪声里；比较 logits、state checksum 和任务指标，才能分辨实现错误与模型能力边界。若需要精确引用，可让 retention 负责连续流摘要，再由少量 attention 或 retrieval 回读原文；这通常比强迫一个固定 state 承担所有可寻址记忆更可控。

## 38.32 小结与资料边界

RetNet 代表论文为 https://arxiv.org/abs/2307.08621。本文使用简化 retention 公式，具体模型中的归一化、多头和 kernel 需要按论文/代码核验。

RetNet 的贡献是把带时间尺度的内容累积、训练并行和在线递推放进统一框架。它的边界同样来自聚合：更稳定的状态成本换取更少的任意历史访问自由度。
