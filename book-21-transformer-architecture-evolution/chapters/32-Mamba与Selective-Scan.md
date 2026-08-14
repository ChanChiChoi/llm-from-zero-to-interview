# 第三十二章：Mamba：Selective State Space 与 Selective Scan

## 32.1 Mamba 针对 SSM 的哪一个弱点

经典 SSM 的参数通常与输入内容无关：状态如何衰减、输入如何写入由固定矩阵决定。这种结构适合连续信号，却可能难以处理语言中的离散选择，例如遇到一个分隔符时应记住后面的实体，遇到新的版本号时应忘掉旧值。

Mamba 的核心方向是让部分状态参数由当前输入产生，使模型可以选择性地传播或遗忘信息。它同时设计 hardware-aware selective scan，在不显式使用 full attention 的情况下完成递归更新。

## 32.2 选择性状态更新的抽象

基础离散 SSM：

~~~math
x_t=\bar A x_{t-1}+\bar B u_t,\qquad y_t=Cx_t+Du_t
~~~

选择性版本可以抽象为：

~~~math
x_t=\bar A_t x_{t-1}+\bar B_t u_t,\qquad
y_t=C_t x_t+D_tu_t
~~~

其中 `\bar A_t,\bar B_t,C_t` 由当前输入或输入投影计算。常见实现会让步长 `\Delta_t`、输入投影和输出投影依赖 token；具体参数化要以 Mamba 版本和代码为准。

如果 `\Delta_t` 较大，某些状态维度可能快速衰减；如果某些输入控制 `B_t` 较强，当前 token 更容易写入状态。这提供了“何时记住、何时忘记”的路径。

## 32.3 小白直觉：可调节的记事本

固定 SSM 像每个 token 都按同一规则写入记事本；Mamba 像记事本上的每一页都有一个门：普通词轻轻带过，关键标记打开门写入，新的事实到来时可以覆盖旧事实。

这并不等于模型能保存所有内容。状态仍有限，门控如果误判，关键 token 可能没有写入；如果不忘旧信息，冲突版本会一起污染状态。

## 32.4 Selective Scan 的计算关系

若状态更新可以写成逐步仿射变换：

~~~math
x_t=A_t x_{t-1}+b_t
~~~

则两个相邻更新可以合并：

~~~math
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,\;A_2b_1+b_2)
~~~

这个结合性让并行 scan 成为可能：训练可以在 chunk 内并行组合更新，推理仍然按单步状态推进。真实 selective scan 还包含向量/矩阵广播、卷积分支、门控和硬件布局。

## 32.5 Mamba block 的教学结构

Mamba block 常被概括为输入投影、局部卷积、选择性 SSM、门控输出和残差。教学化写法：

~~~math
z,x=\operatorname{split}(W_{in}h)
~~~

~~~math
\tilde x=\operatorname{Conv1D}(x)
~~~

~~~math
s_t=F(s_{t-1},\tilde x_t),\qquad
h'=W_{out}(s\odot \sigma(z))
~~~

不同实现的通道数、激活、卷积宽度和归一化可能不同；不要把教学结构当作所有 Mamba 模型的完整配置。

## 32.6 一个 toy selective state

~~~python
import torch


def toy_selective_scan(x, decay_floor=0.05):
    # x: [batch, seq, dim]
    state = torch.zeros_like(x[:, 0])
    outputs = []
    for step in range(x.size(1)):
        token = x[:, step]
        gate = torch.sigmoid(token)
        decay = decay_floor + (1.0 - decay_floor) * gate
        state = decay * state + (1.0 - gate) * token
        outputs.append(state)
    return torch.stack(outputs, dim=1)


y = toy_selective_scan(torch.randn(2, 8, 4))
print(y.shape)
~~~

这个 gate 只是教学示例，更新方向甚至可以与真实 Mamba 不同。重点是输入改变状态保留和写入，而不是某个具体一行公式。

## 32.7 选择性如何改善离散任务

可以设计一个 selective copy 任务：序列里有大量噪声，只有遇到标记 `STORE` 后的值需要在很远处被复制；遇到 `RESET` 后，旧值不应再影响答案。固定衰减模型可能把值逐渐冲淡，选择性 state 可以学会在标记附近改变更新。

评估要测试：标记位置、噪声长度、多个 STORE、RESET 后的旧值、相似实体和精确数字。只测平均分类会掩盖 gate 是否真正执行了记忆控制。

## 32.8 训练、推理和硬件

Mamba 论文强调 hardware-aware selective scan：输入依赖参数破坏了简单的固定卷积，但可以设计融合的 scan kernel，避免把中间状态频繁写回慢速内存。理想情况下，推理只保存固定 state，吞吐随序列长度更可控。

硬件收益取决于实现。小 batch、短序列或 kernel fallback 时，状态更新可能不如成熟 attention；大 batch 长序列、状态可融合时，优势更可能出现。应测 real tokens/s、带宽、occupancy、kernel launch 和 p99。

## 32.9 与 full attention 的能力边界

Mamba 的 state 仍是压缩表示。当前 query 不能像 full attention 那样直接回看任意历史 token；它只能读取已经写入的 state。选择性提高了“写什么”的能力，不会自动提供“从原文任意寻址”的能力。

因此在文档问答、跨段引用和复杂 ICL 中，Mamba 可能需要外部 retrieval、global attention 或 hybrid layer。不同任务的结果不能用单一 Mamba benchmark 外推。

## 32.10 常见失败模式

包括 gate 饱和、state norm 爆炸/消失、卷积 padding 错误、scan 的 chunk 边界不一致、padding token 更新 state、reset 不生效、量化后选择性改变、CPU reference 与 CUDA kernel 不一致。

排查需要记录 gate/step 分布、state norm、长短序列 loss、selective copy exact match、chunk continuation logits 和 kernel 是否 fallback。对服务还要测试 preemption、snapshot、rollback 和 batch 重排。

## 32.11 机制与边界：选择性是 expressivity 与并行的折中

让参数依赖输入增强了内容建模，却破坏了固定卷积核的简单并行。selective scan 的价值在于把输入依赖更新组织成可结合的计算，并根据 GPU 内存层次进行融合。理解这点比背“5 倍吞吐”更重要，因为速度数字依赖版本、硬件、batch 和比较 baseline。

Mamba 的结构通常还包含短卷积和门控分支，它们共同提供局部模式、状态选择和非线性。把它简单称为“没有 attention 的 RNN”会丢掉实现和训练上的关键细节。

## 32.12 面试追问、误区与练习

**问：Mamba 的 selective 解决了什么？**

标准回答：让状态更新的步长、写入和读取部分随输入变化，使模型能选择性保留/遗忘信息，改善固定参数 SSM 在离散内容任务上的表达；它仍然是压缩状态，不等于任意历史检索。

**问：为什么需要 selective scan kernel？**

标准回答：输入依赖参数使固定卷积不再直接适用，需要在 GPU 上高效组织递归/scan，减少中间状态读写并利用并行和内存层次。

常见误区包括把 Mamba 说成固定滑窗、把 selective state 当显式 KV、把论文 throughput 跨硬件复制，以及忽略 state 生命周期。

练习：实现 selective copy/reset 数据集，比较固定衰减、toy selective state 和 attention 的精确复制、长度外推、状态大小和吞吐。

## 32.13 Selective 的真正含义

固定 SSM 对所有 token 使用同一转移规律，Mamba 类设计让参数或状态更新依赖输入，使模型可以选择保留、遗忘或强调信息。选择性并不等于拥有 full attention 的随机访问，它只是让压缩状态更适应当前输入。

可以把输入相关更新写成：

~~~math
x_t=A_t(x_t)x_{t-1}+B_t(x_t)u_t,\qquad
y_t=C_t(x_t)x_t
~~~

这是理解选择性的抽象形式，具体参数化和扫描 kernel 需以论文实现为准。

## 32.14 scan 的正确性和硬件效率

Selective scan 的收益取决于 fused kernel 是否减少中间状态读写。测试需要比较 reference scan、PyTorch 组合实现和优化 kernel 的数值误差、吞吐、显存和不同 batch/长度下的 p99。

长序列中还要检查 chunk 边界、state continuation、padding 和请求取消。一个 kernel 在整齐 batch 上很快，不代表混合长度在线服务也快。

## 32.15 selective 的行为探针

要验证选择性状态是否真的学会了“遇到什么就保留什么”，可以构造带有控制 token 的合成任务。输入包含 STORE、NOISE、RESET 和 QUERY：STORE 后写入一个值，NOISE 只增加长度，RESET 要求忘掉旧值，QUERY 要求输出当前值。

如果固定 SSM 在 RESET 后仍然输出旧值，而 selective 模型能在不同干扰长度下正确切换，说明输入依赖更新提供了功能性收益。还要加入相同表面格式但不同语义的 token，防止模型只记住位置或字符模式。

记录四类指标：STORE 后召回率、RESET 后旧值误报率、干扰长度曲线和 state norm。选择性不是只看 attention map，而是看写入、保持、删除和读取是否随输入改变。

## 32.16 selective scan 的代数和工程边界

一段线性状态更新可以抽象成 affine transform：

~~~math
s_t=A_t s_{t-1}+b_t
~~~

若把变换表示为对 A 和 b 的组合，两个连续片段可以按顺序合并：

~~~math
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,\ A_2b_1+b_2)
~~~

这个结合结构为并行 scan 和 chunk 组合提供了基础，但实际 Mamba block 还有输入投影、卷积、门控、归一化和有限精度。实现不能因为 affine 变换可结合，就假设完整 block 的所有操作都可以任意重排。

## 32.17 新状态模型的判断清单

面对一个声称使用 selective scan 的新模型，先检查公开资料是否说明 state 的 shape、输入依赖参数、训练路径、推理路径和 kernel。再做 full、chunk、step、snapshot restore 四路对照。若只公开“线性复杂度”而没有 state contract，性能和能力都只能作为待验证假设。

服务侧还要测 batch reorder、padding、preemption、speculative rollback 和跨 revision 复用。选择性状态使模型更能处理内容，却也使状态更依赖 token 顺序，错误恢复的影响更大。

## 32.18 用合成任务观察 selective 是否真的工作

只看语言 loss 很难知道 gate 在记忆什么。可以构造四类探针：`STORE` 后远距复制、`RESET` 后旧值清除、多个版本号中选择最新、相同表面 token 在不同控制标记下采取不同记忆策略。每类任务都改变干扰长度、实体数量和标记位置。

记录 store recall、reset false positive、版本选择准确率、state norm、gate/step 分布和 chunk continuation 误差。若模型只在标记出现在固定位置时成功，说明它可能学到了位置捷径；把控制 token 和干扰 token 做词面替换后仍成功，才更能说明选择性来自输入条件。

## 32.19 Selective Scan 的训练—服务边界

训练时可以用 chunk 内并行 scan，推理时通常保存递归 state。两条路径必须对齐初始状态、padding、reset、位置步长和有限精度。服务端还要在 batch reorder、preemption、取消、snapshot restore 和多租户隔离时保存 state owner 与 step。

硬件 benchmark 应把 reference loop、融合 kernel、不同长度 batch 和 fallback 路径放在同一表中。对短序列或小 batch，kernel launch 和状态读写可能抵消理论优势；对长流，固定 state 可能改善显存，但 state snapshot 和迁移也会成为新的工程成本。

## 32.20 state checkpoint 与 batch reorder

Selective state 不是一个可以按请求结束就随意复制的普通张量。服务端至少要绑定 request、model revision、logical position、dtype、state schema 和 owner worker。batch 重排时，state 必须跟着 request 一起移动；padding、reset 和 chunk continuation 不能只改变 token batch 而不改变 state mask。

抢占或迁移时可以选择保存 state、重算最近 chunk 或直接拒绝迁移。保存 state 的成本较低，但对版本和数值误差敏感；重算更容易验证，却增加延迟和 GPU 计算。恢复后要用同一后缀比较 logits 或 token，并检查 state checksum，而不是只看任务最终是否输出文本。

## 32.21 Selective state 是一种控制策略

Selective scan 的关键变化不是把普通 SSM 的状态维度简单做大，而是让输入参与决定当前信息应该写入、保留还是衰减。这个变化可以看成一个隐式的控制策略：同样的历史在不同输入条件下，更新路径可能不同。因此它提升了内容依赖能力，也让状态更难用一个固定公式解释。

对专家而言，至少要区分三类量：状态转移、输入到状态的注入，以及状态到输出的读取。一个 gate 看起来有变化，不代表它真的改变了任务决策；要做反事实，把 gate 固定为均值、全开或随机，再看 exact recall、reset 错误和状态范数是否变化。若 gate 只在训练数据的固定标记位置变化，可能是位置捷径而不是语义选择。

## 32.22 Selective scan 的手算验收

构造三个 token 序列：第一组在开头写入数字后持续加入干扰，第二组在中间加入 `RESET`，第三组在相同词面下用控制 token 改变是否保存。对每个序列比较完整 scan、逐 token scan 和 chunk continuation 的 state 与输出。

验收至少记录 `store_recall`、`reset_false_positive`、`conditional_recall`、`state_norm` 和 `max_logit_error`。如果 reset 后仍然能回答旧数字，说明 gate 没有清除旧状态或读取路径仍然泄漏；如果完整 scan 与逐 token 路径不一致，先查 scan combine 和浮点累计；如果只在长序列失败，再区分状态容量、数值漂移和训练长度。

## 32.23 Selective Scan 的训练与服务分工

Mamba 原始论文为 https://arxiv.org/abs/2312.00752。本文用抽象更新解释选择性和 scan，具体 Mamba block、参数生成、卷积分支和 kernel 应查对应版本源码/技术报告。

Mamba 的核心不是一句“线性复杂度”，而是把输入依赖的记忆控制和硬件友好的状态扫描结合起来。它改善了状态模型的内容选择能力，却仍需面对精确检索和生态边界。

## 32.24 Selective scan 的代数与并行化

对一个简化的仿射状态更新：

```math
s_t=A_t s_{t-1}+b_t,
```

可以把一段 token 的变换表示成二元组 `(A,b)`。连续两段的组合为：

```math
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1, A_2b_1+b_2).
```

这个组合满足结合律，因此理论上可以用 scan 在训练时并行计算多个位置。注意，结合律只说明“状态转移的组合”可重排，不说明完整 Mamba block 的卷积、投影、门控、归一化和残差可以随意重排。实现要先把可结合的部分和不可结合的部分分开，才能判断 kernel 是否等价。

验证时可以用随机 `A_t,b_t` 比较串行 reference、chunk scan 和 tree/parallel scan 的 state。若组合结果在短序列一致、长序列漂移，优先查乘法顺序、累积精度、chunk 边界和 state dtype，而不是先归因于模型表达能力。

## 32.25 选择性门控的反事实解释

“输入决定记忆”需要反事实证据。给定同一输入序列，分别运行真实 gate、固定均值 gate、全开 gate、随机 gate，并比较 store recall、reset false positive、版本选择、state norm 和任务输出。若真实 gate 与固定 gate 的结果没有差异，说明 gate 可能只是参数化噪声；若只在固定控制 token 位置有差异，还要排除位置捷径。

可以把 gate 对状态的影响写成局部敏感度：

```math
\Delta_t
=\left\|s_t(g_t+\delta)-s_t(g_t)\right\|.
```

敏感度高不等于有益，可能只是数值放大。要把 `\Delta_t` 与正确的实体召回、reset 行为和安全任务一起看；如果 gate 改变了 state，却没有改善可验证任务，不能把激活变化直接解释为“模型学会了选择性记忆”。

## 32.26 Mamba 的状态成本与 attention 的 KV 成本

对每个请求，attention 通常保留与历史 token 数相关的 KV；Mamba 类路径保留固定或相对固定的递归 state。粗略地写：

```math
M_{\mathrm{KV}}\propto BTLn_{kv}d_h,
\qquad
M_{\mathrm{Mamba\ state}}\propto BLm d_s.
```

这不是质量等式，也不表示 Mamba 完全没有中间 activation、卷积缓存和 workspace。它说明两种路径的资源增长不同。高并发短请求可能被 kernel launch 和 state 管理主导；超长流可能从固定 state 获益；需要任意历史引用的任务则可能增加 retrieval 或 dense fallback，重新引入外部成本。

公平 benchmark 要锁定模型参数、tokenizer、训练长度、batch、dtype、硬件、输出协议和质量验收条件，再报告 prefill、decode、state/KV bytes、p99、迁移恢复和 fallback。只用理论内存公式预测线上体验是不够的。

## 32.27 Selective state 的生产边界

选择性状态适合持续流、长序列和需要压缩上下文的路径，但生产系统仍要回答五个问题：状态能否精确快照，模型升级后能否恢复，跨 worker 迁移是否安全，证据是否可回读，外部工具是否只在验证后执行。任何一个问题没有答案，模型就只能作为受限候选路径，而不能无条件替代显式上下文。

一个合理的混合设计可以让 Mamba 扫描长流，检测器触发候选区间，retrieval 返回原始证据，decoder/attention 完成精确回答。此时 Mamba 输出的是召回信号或摘要，不是事实来源。trace 要记录 state revision、候选区间、检索来源、最终引用和 fallback；这样发生错误时，团队能判断是选择性写入、候选触发还是最终读取出了问题。

Mamba 的价值因此不在于“所有 attention 都应该消失”，而在于提供一种可扩展的状态计算路径。它是否适合一个任务，取决于历史访问模式、精确性要求、硬件 kernel、服务恢复和外部证据边界的共同结果。
