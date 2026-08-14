# 第三十五章：Mamba-2 与 State-Space Duality

## 35.1 为什么需要重新看 SSM 与 Attention 的关系

Mamba 的选择性 state 让 SSM 更适合离散序列，但训练时的 scan、硬件利用和模型扩展仍有优化空间。Mamba-2 相关工作从 state space duality 出发，说明某些结构化状态更新可以与一种受约束的 attention/半可分离矩阵计算互相转换。

这里的“dual”不是说 SSM 和标准 attention 完全等价，也不是把所有 attention 都改成 SSM。它提供了一个分析工具：同一组序列变换可以从递归状态、卷积核或结构化矩阵三个角度表达，从而选择更适合训练/推理的计算路径。

## 35.2 从递归到半可分离矩阵

考虑标量或对角状态更新：

~~~math
s_t=a_t\odot s_{t-1}+b_t\odot u_t
~~~

展开后，输出在位置 t 对位置 i 的影响可以写成：

~~~math
y_t=\sum_{i\le t}c_t\odot
\left(\prod_{j=i+1}^{t}a_j\right)
\odot b_i\odot u_i
~~~

对应的 token-token 权重不是任意矩阵，而具有由前缀乘积构成的结构。它可以被视作半可分离（semiseparable）矩阵：对角线附近和低秩/结构化部分容易表示，完整自由度低于标准 attention。

## 35.3 State Space Duality 的三种视角

同一层可以从三个角度理解：

1. **递归视角**：每个 token 更新 state，适合单步 decode。
2. **卷积视角**：线性时不变或局部近似下生成 kernel，适合并行训练。
3. **结构化矩阵视角**：用半可分离/扫描结构表达所有位置间的影响，寻找高效并行算法。

选择视角取决于阶段。训练希望高并行、较大矩阵和少写回；推理希望固定 state、低内存和低启动开销。dual 的价值在于让研究者不必把一种实现路径写死。

## 35.4 与标准 attention 的差异

标准 attention 的权重为：

~~~math
A_{ti}=\operatorname{softmax}(q_t^\top k_i)
~~~

每个 query 都可以根据内容独立改变对历史的排序。SSM dual 产生的权重通常受 state transition、输入选择和结构化前缀乘积约束：

~~~math
W_{ti}=c_t\odot
\left(\prod_{j=i+1}^{t}a_j\right)
\odot b_i
~~~

它可以表达输入依赖记忆，却未必拥有任意 `T×T` 内容相似矩阵的自由度。这种限制换来了状态和计算结构，不能用“等价 attention”一句话抹平。

## 35.5 Chunked state-space duality

长序列训练可以分成 chunk。每个 chunk 内使用并行结构计算局部输出，同时保存一个边界 state；下一个 chunk 以该 state 为初始值。设第 c 个 chunk 的状态变换为：

~~~math
s_{\mathrm{out}}^{(c)}
=A^{(c)}s_{\mathrm{in}}^{(c)}+b^{(c)}
~~~

跨 chunk 可以用仿射变换合并，局部计算则能在 GPU 上并行。chunk size 决定并行度、workspace 和边界通信；太小会增加调度，太大则减少流式性。

## 35.6 一个教学版 chunk scan

~~~python
def affine_compose(left, right):
    # 每个变换表示 s_out = a * s_in + b
    a1, b1 = left
    a2, b2 = right
    return a2 * a1, a2 * b1 + b2


def run_chunks(values, decay, chunk_size):
    state = 0.0
    outputs = []
    for start in range(0, len(values), chunk_size):
        for value in values[start:start + chunk_size]:
            state = decay * state + (1.0 - decay) * value
            outputs.append(state)
    return outputs


print(run_chunks([1, 0, 0, 2, 0, 0], 0.9, chunk_size=2))
~~~

这段代码仍按 token 循环，目的是展示 chunk 边界不应重置 state。真正 SSD/Mamba-2 实现会在 chunk 内采用更高效的矩阵/scan 算法。

## 35.7 为什么训练 kernel 需要重新设计

如果把每个状态维度逐 token 更新，GPU 可能处于低 occupancy；如果把所有 token 展成 dense 关系矩阵，又失去 SSM 的内存优势。结构化矩阵算法试图在两者之间找到并行块：减少不必要的中间写回，让 state transition 和输入投影尽量融合。

最终性能取决于 state dimension、chunk、batch、dtype、GPU 架构和 kernel 实现。论文中的 FLOPs 下降不能直接换算成服务 latency。

## 35.8 Mamba-2 对系统设计的启发

一个层可能同时需要：训练路径的并行 kernel、推理路径的递归 state、checkpoint 中的参数结构、state snapshot 格式和不同长度的调度。若只实现 forward，不实现 state continuation，模型无法可靠在线化。

服务端可按请求类型选择：短输入大 batch 使用并行 prefill，长流使用递归 decode；暂停、迁移和恢复时保存 chunk 边界 state 以及 model revision、dtype、position contract。

## 35.9 与 hybrid attention 的关系

SSM dual 不会消除显式 attention 的价值。混合模型可以让大多数层使用结构化 state，少数层承担任意内容寻址；也可以在多模态 token、工具结果或文档边界打开 global path。这样可以把最昂贵的自由度用在高价值位置。

但混合增加了 schedule、cache 和评测矩阵。要知道是 SSM 负责局部趋势，attention 负责精确证据，还是二者都在重复同一工作。

## 35.10 常见失败模式

包括把递归/卷积/结构化矩阵三种公式当成无条件等价；chunk 边界错误；state transition 复合顺序颠倒；训练和推理使用不同 `A/B/C` 参数；scan kernel 对短序列反而慢；以及把 Mamba-2 的研究结构推断到没有公开细节的模型。

排查先做标量 reference：逐步递归与 chunk 组合输出一致；再验证向量/矩阵 kernel；最后测不同 chunk、batch、长度和 dtype。对外部模型只写公开资料能支持的结构。

## 35.11 机制与边界：半可分离结构的能力边界

半可分离矩阵为序列影响提供结构化自由度。它比单一固定卷积更灵活，比任意 dense attention 更受约束。理解其秩、状态维度、输入依赖和层间组合，有助于解释为什么多层 SSM 仍能形成复杂长程模式，也有助于识别哪些查询可能造成状态碰撞。

研究比较应报告参数、state size、chunk size、训练/推理路径、kernel 和任务级证据。不要把“dual”当成性能保证，也不要把结构化表示的表达能力等同于 full attention。

## 35.12 面试追问、误区与练习

**问：State-Space Duality 说明 SSM 等于 Transformer 吗？**

标准回答：不等于。它说明某些状态更新可以与结构化的 attention-like 矩阵表达互换，帮助设计训练/推理算法；标准 attention 的任意内容寻址自由度和 SSM 的递归约束仍然不同。

**问：为什么要做 chunked scan？**

标准回答：在保持跨 chunk state continuity 的同时，把局部计算并行化，平衡 GPU 利用率、workspace、边界通信和流式恢复成本。

常见误区包括把每个状态模型都叫 SSD、忽略合并顺序、只看理论复杂度，以及不验证 chunk continuation。

练习：实现标量 affine transform scan，比较逐步递归、chunk 内并行抽象和错误重置 state 三种结果，扩展到不同 chunk size。

## 35.13 Duality 为什么重要

State-Space Duality 的价值在于把看似不同的 attention-like 计算和状态空间递推放到一个可互相转换的视角中。训练可以使用更适合 GPU 的矩阵或卷积形式，推理可以使用递归 state；但转换成立需要满足具体代数条件。

因此读论文时要同时看 parallel form、recurrent form、chunk form 和边界 state 的定义。只记住“二者等价”而忽略初始化、归一化和有限精度，会导致实现错误。

## 35.14 从等价计算到工程选择

若 parallel form 适合 prefill，recurrent form 适合 decode，chunk form 可能是折中。真实系统应按阶段选择 kernel，并验证三种路径的 logits 一致性：

~~~math
\left\|y_{\mathrm{parallel}}-y_{\mathrm{chunk}}\right\|
\le \epsilon,\qquad
\left\|y_{\mathrm{chunk}}-y_{\mathrm{recurrent}}\right\|
\le \epsilon
~~~

误差门限要和任务指标绑定，不能只看张量平均误差。

## 35.15 duality 的代数条件和边界

State-Space Duality 连接的是满足特定结构条件的状态更新与矩阵化序列计算，不是“任意 RNN 都等于 attention”。当输入投影、状态转移、归一化或门控改变时，等价关系可能需要重新推导。有限长度、初始 state、padding 和数值精度也会让理论等价变成近似等价。

阅读一个 duality 结果时，可以依次核对：矩阵的下三角因果结构是什么，状态合并是否满足结合律，输入和输出投影在哪里，chunk 边界携带哪些信息，以及实现是否保留了同一归一化。少看一个条件，就可能把论文中的某个特例误写成普遍定律。

## 35.16 chunk size 是算法和硬件的共同参数

chunk 越大，矩阵运算越容易获得 GPU 并行，单次 kernel 的固定开销占比下降；但 workspace、边界延迟和抢占粒度都会上升。chunk 越小，更接近在线流式执行，状态传递更频繁，可能受 launch 和同步影响。

设每个 chunk 的计算时间为 t_c、边界传递时间为 t_s、chunk 数为 N_c，粗略端到端时间是：

~~~math
T\approx N_c t_c+(N_c-1)t_s
~~~

真实系统还会受 batch、负载不均、通信和 cache 命中影响。压测时应把 chunk size 作为一等变量，同时记录吞吐、p99、workspace、恢复时间和输出误差。

## 35.17 从研究公式到 serving engine

要把 Mamba-2 类结构放进 serving engine，至少要实现四个对象：参数权重、parallel kernel、recurrent state 和 state transition 的版本契约。scheduler 还要知道请求处于 prefill、decode、paused 还是 rollback 状态，并在 batch 重排时移动对应 state。

一个可靠的升级路径是先用标量 reference 验证递归和 chunk 组合，再用向量实现替换，最后接入 fused kernel 和量化。每一步都保留同一组 golden logits 和 state checksum。这样性能优化不会把“理论等价”变成无法定位的线上差异。

## 35.18 Duality 不等于能力完全等价

State-Space Duality 说明某些状态空间计算可以改写成半可分离或类似 attention 的矩阵计算，从而帮助训练 kernel 设计；它不意味着 SSM 与标准 softmax attention 在任意输入、参数和任务上完全相同。中间的参数化、归一化、因果约束、门控和数值误差都决定了实际函数族。

阅读公式时要区分三种“等价”：代数重排等价、在特定参数下的数值近似等价、在任务指标上的经验相近。只有第一种可以直接由推导证明；后两种必须通过实验支持。

## 35.19 Chunk size 同时影响算法和硬件

chunk 太小，边界通信、kernel launch 和状态合并开销上升；chunk 太大，片上内存不足、激活占用和长序列尾延迟上升。更重要的是，chunk 内的并行计算和 chunk 间的递归状态必须保持相同的因果顺序。

应对多个 chunk size 测 full、chunked、step 三路输出误差，并记录 tokens/s、峰值显存、通信时间和 p99。在线服务还要测不同请求长度混合、取消和 preemption；论文中的最佳 chunk 不一定是生产 batch 的最佳 chunk。

## 35.20 数值误差、边界 state 与恢复

parallel、chunk 和 recurrent 三条路径在有限精度下通常只能近似一致。误差会在长序列的状态合并中累积，尤其当衰减、归一化或门控接近极端值时。验证不应只比较最终 logits 的平均误差，还要比较 top-k、停止 token、长流状态和任务级 exact recall。

checkpoint 要说明保存的是 chunk 边界 state、完整 activation 还是可以重算的输入引用。若只保存边界 state，必须绑定 chunk size、参数和归一化版本；若 chunk size 变了，旧 state 可能无法解释。迁移时更安全的路径是回到最近的兼容边界重算，并把重复计算记入恢复成本。

## 35.21 用三种等价性做回归

阅读 State-Space Duality 时，可以做三种逐层验证。第一种是代数验证：在小维度上比较递归展开与矩阵形式；第二种是数值验证：在相同 dtype、chunk 和参数下比较 hidden、logits 和 state；第三种是任务验证：比较 exact recall、长流分类和恢复后的输出。

这三层不能互相替代。代数相同但实现可能因 layout、mask 或累计精度产生误差；数值接近但任务可能在决策边界翻转；任务平均分相同也不代表每个位置都一致。对生产 kernel，应保存输入、参数 revision、chunk size、state checksum 和误差分位数，避免只保留最终 benchmark。

## 35.22 SSD 等价的使用边界

State Space Duality 提供的是一组结构联系和可转换的计算路径，不意味着任意 Transformer 都能无损替换成 Mamba-2，也不意味着训练并行路径和在线递归路径在有限精度下自动一致。阅读论文时要区分理论等价、特定参数化下的算法等价和实际 kernel 的近似。

一个最小验证可以对同一组参数和输入比较 full sequence、chunked scan、recurrent update 的 hidden 最大误差，并按长度、dtype、chunk size 和 state reset 分桶。只有误差、任务质量和吞吐都在验收条件内，才能把速度优势写成工程结论。

## 35.23 Mamba-2 与 attention 的职责边界

Mamba-2 的线性状态路径适合流式、长序列和压缩历史；attention 仍擅长按当前 query 选择任意历史 token。对趋势预测和持续音频，两者差异可能很小；对两个同名实体、版本冲突、精确数字和引用任务，必须做内容寻址测试。混合方案的价值不是宣布谁取代谁，而是让状态扫描承担大多数 token，让少量显式路径承担高精度回读。

## 35.24 State-Space Duality 的工程含义

Mamba-2/State-Space Duality 的代表论文为 https://arxiv.org/abs/2405.21060。具体结构、算法和模型配置要以论文版本、官方代码和 checkpoint 为准。

Mamba-2 的重要启发是：架构和实现路径可以双向设计。递归 state、卷积和结构化矩阵各自服务不同阶段，但它们的等价关系有条件，不能替代任务级证据。

## 35.25 Duality 不是无损格式转换

同一组状态更新可以有递归、卷积、块矩阵或 scan 表达，但每种表达依赖因果 mask、参数化、边界 state 和浮点路径。把公式改写成矩阵形式后，kernel 仍可能使用不同的累积顺序、padding 和 truncation；因此“数学上等价”不能直接变成“输出 bitwise 相同”。

工程上应给等价性加条件：相同参数 revision、position、initial state、chunk、dtype、归一化和 mask。条件变化时，结论降级为近似或需要重新实验。

## 35.26 Chunk 与 checkpoint 的联合设计

chunk size 既决定 parallel scan 的效率，也决定可保存的恢复边界。chunk 较大可以减少边界开销，却增加重算和临时激活；chunk 较小便于恢复，却增加 kernel launch、状态合并和 metadata。可用教学近似表示：

```math
C_{\mathrm{total}}(c)
=C_{\mathrm{compute}}(c)+C_{\mathrm{boundary}}(c)
 +C_{\mathrm{checkpoint}}(c)+C_{\mathrm{recompute}}(c).
```

线上应根据故障率、SLO、存储带宽和请求长度选择，而不是直接使用论文默认 chunk。恢复测试要比较同一 suffix 在 full、chunk、snapshot restore 和迁移路径的 logits 与任务结果。

## 35.27 Mamba-2 与 attention 的混合使用

State Space Duality 允许研究者用类似矩阵/attention 的视角设计训练 kernel，但生产系统仍可以让 Mamba-2 负责长流扫描，让少量 attention 或 retrieval 负责精确查询。混合时显式记录两种历史表示和交接位置；不能把 Mamba state 当作完整 K/V，也不能在没有训练交接层的情况下假设它们可互换。

最终评估将三种证据放在一起：代数/数值等价、目标任务质量、真实 runtime 成本。若第一项通过而第三项不通过，问题是 kernel 或调度；若前两项通过而引用失败，问题是状态可寻址性；若 chunk restore 失败，问题是 serving contract。这个分层能把 Mamba-2 的启发转化为可执行判断。
