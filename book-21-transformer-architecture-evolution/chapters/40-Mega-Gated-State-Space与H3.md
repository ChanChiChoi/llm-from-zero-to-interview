# 第四十章：Mega、Gated State Space、H3 等早期混合路线

## 40.1 为什么这些路线值得学习

Mamba 之前，研究者已经在尝试把 attention、门控、卷积和状态空间结合起来。Mega 使用移动平均和 gated attention，H3 研究结构化状态空间、短卷积和门控的组合，Gated State Space 则让状态更新拥有输入相关的选择。

它们没有都成为通用 LLM 的默认架构，但提供了重要的设计谱系：纯递归状态容易缺少内容寻址，纯 attention 成本高，门控/局部卷积/多尺度状态可以承担不同信息流。

## 40.2 Mega：平滑记忆加内容门控

移动平均状态可以抽象为：

~~~math
m_t=\alpha_t\odot m_{t-1}+(1-\alpha_t)\odot x_t
~~~

再把 query/key/value 的 gated attention 与 m_t 组合：

~~~math
y_t=g_t\odot \operatorname{Attn}(x_{\le t})
(1-g_t)\odot \operatorname{Mix}(m_t)
~~~

Mega 的实际结构包含更具体的 EMA、gated attention、分解和 normalization；上式只说明两个信息源：平滑时间状态和内容相关 attention。

## 40.3 小白直觉：趋势滤波器加临时放大镜

移动平均像记录总体趋势，gated attention 像临时拿放大镜查看相关片段。趋势适合稳定噪声，attention 适合回答某个细节。gate 决定当前输出更依赖哪一条路径。

如果 gate 永远偏向趋势，精确检索会变差；如果永远偏向 attention，计算优势消失。gate 的分布和任务分桶结果需要一起监控。

## 40.4 H3：状态、短卷积与门控的组合

H3 类路线可以用三个模块直觉理解：用状态空间组件传播长程信息，用短卷积捕捉局部模式，用门控让不同路径相互调制。一个抽象表达为：

~~~math
z_t=\operatorname{SSM}(x_t),\qquad
c_t=\operatorname{Conv}(x_t)
~~~

~~~math
y_t=W_o\left(g_t\odot z_t+(1-g_t)\odot c_t\right)
~~~

真实 H3 结构和参数化更具体，不能由这个公式推断完整模型。学习价值在于看到局部、长程和选择性并不是互斥模块。

## 40.5 门控状态空间的基本方程

一般状态更新为：

~~~math
s_t=A s_{t-1}+B x_t
~~~

门控版本可以写成：

~~~math
s_t=g_t\odot(A s_{t-1})+(1-g_t)\odot(Bx_t)
~~~

或者让 A/B/C 由输入决定。不同位置放 gate 会改变语义：放在旧状态上是遗忘控制，放在输入上是写入控制，放在输出上是读取混合。面试时应说明 gate 的位置，而不是只说“有门控”。

## 40.6 一个路径混合 demo

~~~python
import torch


def gated_state_conv(x, decay=0.9):
    # x: [batch, seq, dim]
    state = torch.zeros_like(x[:, 0])
    previous = torch.zeros_like(state)
    outputs = []
    for step in range(x.size(1)):
        local = x[:, step] - previous
        state = decay * state + (1.0 - decay) * x[:, step]
        gate = torch.sigmoid(x[:, step])
        outputs.append(gate * state + (1.0 - gate) * local)
        previous = x[:, step]
    return torch.stack(outputs, dim=1)


print(gated_state_conv(torch.randn(2, 6, 4)).shape)
~~~

这里的 local 分支只是相邻差分，不是 H3/Mega 的实现；它让读者看到 state path 和 local path 的门控组合。

## 40.7 为什么混合通常比纯单一路线更现实

真实序列同时包含局部语法、趋势、事件、远距引用和噪声。单一固定窗口可能丢趋势，单一 state 可能丢细节，单一 full attention 成本高。混合模块可以分配职责。

但混合也带来参数、归一化、gate 训练和 kernel 复杂度。若路径没有学出分工，模型只是重复计算；如果 gate 饱和，某一路径形同虚设。

## 40.8 训练和推理取舍

EMA/SSM/卷积可以在线更新，gated attention 仍可能需要 KV 或局部 cache。训练时路径可并行，推理时需要保存不同类型的状态。服务端必须知道每个层的 cache schema，不能用同一个 KV manager 假设覆盖全部路径。

当 local/state 路径占多数、attention 只在少数层使用时，可能降低长流成本；但 global 层仍会产生随长度增长的 cache，除非配合窗口、MLA 或检索。

## 40.9 失败模式

包括 gate 饱和、两个分支尺度不匹配、state/conv padding 不一致、local 分支误读未来、attention 分支 cache 未回收、训练时混合比例和推理时不同、以及某个分支 kernel fallback。

应记录 gate histogram、分支输出 norm、各分支梯度、不同长度质量、KV/state bytes、p99 和分支消融。把某一路径置零是最直接的分工诊断。

## 40.10 机制与边界：从模块组合到信息路由

混合架构的研究重点不是“加更多模块”，而是决定信息在模块之间的路由。gate 可以是 token/channel/head/layer 级；动态 gate 增强表达却可能造成不规则 kernel，固定 schedule 容易部署却缺少适应性。

对专家比较要看：状态 transition 的结构、卷积感受野、attention 入口、gate 参数化、训练稳定、路径消融、长程证据和硬件。Mega/H3 是历史路线，不能把它们的所有机制归到 Mamba 或反之。

## 40.11 面试追问、误区与练习

**问：为什么早期混合路线没有马上成为通用 LLM 标准？**

标准回答：它们需要在大规模语言训练、内容寻址、kernel、生态和服务状态之间同时过关；部分路线在特定任务有优势，但通用 LLM 需要更强证据和更成熟实现。

**问：混合模型如何证明模块真的分工？**

标准回答：做路径消融、gate 分布、局部/远距/精确引用分桶，以及分别测各路径计算、状态和 p99；不能只看总 loss。

常见误区包括把所有门控都叫 attention、把移动平均当完整 SSM、只看模块名，以及忽略不同路径的 cache 生命周期。

练习：实现一个 local+state+attention toy block，分别关闭三条路径，设计任务证明每条路径的增量价值。

## 40.12 门控、平滑和短卷积的分工

Mega 的平滑状态适合提取长期趋势，门控决定当前输入是否应改变状态；H3 组合状态、短卷积和门控，让局部模式与长程记忆同时存在。三者的作用不应被概括成“都是更快的 attention”。

做路径消融时分别关闭平滑、短卷积和 gate，测局部语法、长程复制、突发事件和连续流。这样才能知道收益来自局部滤波、状态容量还是输入选择。

## 40.13 复杂模块的 serving 代价

多路径模块可能需要多个 state、不同 kernel 和额外 workspace。在线服务要记录每条路径的 state bytes、update time、fallback、batch 利用率和恢复语义。理论线性复杂度不能替代这些指标。

## 40.14 门控塌缩的诊断

门控路径容易出现两种塌缩：所有输入都使用状态路径，或者所有输入都依赖局部/卷积路径。应查看 gate histogram、不同任务的路径比例、梯度和固定 gate 消融。

如果门控全开但质量没有提高，可能只是冗余计算；如果门控全关但长程 recall 下降，说明模型没有学会按任务分工。正则化要服务于任务质量，不能只让分布看起来均匀。

## 40.15 复杂状态的版本化

Mega/H3 类模型可能有多个状态、短卷积缓存和门控统计。snapshot 需要保存它们的顺序、dtype、step、位置和 owner；升级 kernel 或 layer schedule 时，旧状态应拒绝复用并触发重算。

## 40.16 门控路径的可辨识性

一个混合 block 即使包含 state、短卷积和门控，也不代表每条路径都贡献了能力。若两条路径都能产生相似的平滑表示，优化器可能让其中一条承担全部工作，另一条只消耗参数和显存。要判断分工是否真实，至少需要做三种干预：

1. 把某条路径的输出置零，观察任务和梯度变化；
2. 固定 gate，只训练两条路径，观察是否仍能互补；
3. 交换局部任务与长程任务，检查不同路径的响应是否随任务变化。

可以把路径增量写成：

~~~math
\Delta Q_p=Q_{\mathrm{full}}-Q_{\mathrm{without}\ p}
~~~

其中 p 是某条路径，Q 可以是 exact recall、局部准确率或单位成功成本。若所有路径的增量都接近零，可能是测试不够敏感；若只有一条路径有效，则混合结构可能需要简化或重新训练。

## 40.17 state、短卷积与 attention 的容量预算

不同路径的内存不能只按参数量估算。设每个请求的 SSM state 为 S，短卷积 overlap 为 W，attention KV 为 K，若层数和 dtype 已折算到字节数，则每请求工作集近似为：

~~~math
M_{\mathrm{request}}
\approx M_{\mathrm{state}}(S)+M_{\mathrm{conv}}(W)+M_{\mathrm{KV}}(K)
+M_{\mathrm{workspace}}
~~~

长流 workload 可能由 state 数量主导，短 prompt 大 batch 可能由 workspace 主导，工具调用和精确引用又会扩大 attention/KV 部分。容量规划应按 prompt length、decode length、batch、抢占率和恢复比例分桶，而不是用单一平均请求。

这也是为什么“少量 attention 层”不是免费的。即使层数很少，只要它们保留显式 KV，极长请求仍可能占据主要显存；如果改成窗口或压缩 cache，又要重新评估引用能力。

## 40.18 从实验到服务验收条件

混合模块上线前可以设置三道门：第一道是路径数值一致性，full、chunk 和 streaming 的输出误差必须在阈值内；第二道是行为验收条件，局部、长程、冲突和恢复任务分别达标；第三道是系统验收条件，state bytes、p99、batch 利用率和故障恢复时间满足 SLO。

若某一条路径 kernel 不可用而退回通用实现，系统应记录 fallback，而不是继续使用离线 profile 作为线上数据。模型版本升级时，gate 分布、状态 schema 和路径消融也应进入回归集。

## 40.19 门控到底改变了哪条信息路径

门控可以作用在输入写入、状态衰减、输出读取或残差融合上。它们看起来都叫 gate，但行为不同：写入 gate 决定当前 token 是否进入状态，衰减 gate 决定旧状态保留多久，读取 gate 决定当前 query 读多少状态，残差 gate 决定新分支如何影响主干。

分析一个新门控模块时，先画出四条路径，再做控制变量实验。固定参数和 kernel，只关闭某一种 gate，测 selective copy、reset、长流预测、state norm 和梯度稳定性。这样才能知道收益来自内容选择、数值稳定，还是单纯增加了非线性和参数。

## 40.20 门控状态的数值和服务边界

门控饱和会让状态几乎不更新或几乎不遗忘，低精度可能进一步改变阈值附近的行为。训练中应记录 gate 分布、state norm、梯度和不同长度 loss；推理中还要测试 chunk、padding、reset、batch reorder 和 state restore。

如果门控状态用于在线服务，每个请求必须有独立 owner、step、revision 和取消释放逻辑。门控并不自动提供原文回读能力，涉及引用、权限和删除的任务仍需要 artifact 或检索。

## 40.21 gate 的稳定性与容量账本

门控可以抽象成：

```math
g_t=\sigma(W_gx_t),
\qquad
s_t=g_t\odot\widetilde{s}_t+(1-g_t)\odot s_{t-1}.
```

当 gate 长期接近 0，状态几乎不更新；长期接近 1，则可能放大噪声或遗忘长期信息。训练和服务都要记录 gate 分布、state norm、梯度/NaN、不同长度的召回和 reset 行为。门控的额外投影、卷积和状态读写也要进入每 token 计算与显存账本。

## 40.22 门控到底有没有学到分工

门控路径的存在不等于门控产生了有用分工。可以记录不同任务、层、时间位置和输入类型下的 gate 分布，并做反事实实验：固定输入和主干，只把 gate 替换为均值、全开或随机值，观察质量、状态范数和计算量变化。

如果 gate 长期饱和在 0 或 1，复杂结构可能退化成更简单的子网络；如果 gate 对输入极其敏感，量化、batch 组成和微小格式变化可能带来输出不稳定。报告应给出 gate histogram、饱和比例、层间相关性和任务切片，而不是只说“加入门控后更强”。

## 40.23 混合模块的消融顺序

Mega、Gated State Space 和 H3 这类路线通常组合平滑、门控、短卷积和状态。为了判断收益来自哪里，可以按单变量顺序做：基础状态；加平滑；加门控；加短卷积；加多尺度；最后再测完整模块。每次保持训练预算、参数规模和 serving batch 尽量一致。

系统指标也要一起记录。门控可能增加逐元素计算，短卷积可能带来 state buffer，复杂状态可能增加 snapshot 和量化成本。若质量提升主要出现在短任务，而长任务或 p99 退化，工程选择应考虑混合路径是否值得，而不是只比较平均 benchmark。

## 40.24 状态版本化和回退

复杂混合模块的 state 不是一个通用 tensor。它可能包含平滑累计、短卷积窗口、门控缓存和层位置。模型 revision、dtype、chunk boundary、reset 标记和 batch slot 都属于 state schema。升级时旧 state 不能自动交给新模块解释。

线上切换可以选择排空请求、双版本 shadow、按 schema 双读双写或直接关闭状态复用。若模块失败，target-only 或 attention fallback 必须从最近一次已提交状态恢复，不能把未验证的 gate/conv 更新继续传下去。

## 40.25 证据范围与研究定位

早期混合路线的论文可以支持模块动机、公开方程和实验设置；“更适合生产”需要目标硬件、kernel、长上下文、状态恢复和任务质量的实测。不同实现对 Mega、H3、gated SSM 的命名和组合并不相同，不能把相似名称当作相同模型。

学习这些路线的价值，在于理解信息路径和工程约束如何共同塑造架构，而不是预言某个模块一定替代 Transformer。成熟结论应同时写出收益、代价、失败模式和可回退方案。

## 40.26 门控状态的质量—稳定性曲线

门控越灵活，输入依赖越强，但对量化、格式变化和 batch 组成也可能越敏感。可以沿着 gate temperature、精度、序列长度和干扰强度做扫描，记录 gate 饱和率、状态范数、exact recall、长流漂移和 kernel time。若 gate 只在训练分布内有效，离开固定模板后性能下降，说明模型学到的是协议捷径。

对于 Mega/H3 类混合模块，先测试每个子路径，再测试组合；若移除短卷积质量不变而减少 state bytes，说明它可能不是目标 workload 的必要组件；若移除门控后长程任务失败，才有证据说明门控承担了选择性记忆。这样的消融比直接引用完整模型分数更有解释力。

## 40.27 从门控方程看“选择性记忆”到底是什么

把门控状态写成最小形式，有助于区分传播、写入和读取三个动作：

~~~math
s_t=\lambda_t\odot s_{t-1}+u_t,
\qquad
u_t=b_t\odot\phi(x_t),
\qquad
y_t=r_t\odot\psi(s_t).
~~~

`lambda_t` 决定旧状态保留多少，`b_t` 决定当前输入写入多少，`r_t` 决定哪些状态通道暴露给输出。Mega、gated SSM 或 H3 的具体方程可能不同，但工程分析都应问这三个问题：输入什么时候被写入，旧信息什么时候被遗忘，query 如何读出。只看到一个 gate tensor 的均值，无法说明模型真的学到了有用分工。

例如，在“事件发生后保持状态”的任务中，理想 gate 会在事件位置提高写入，在无关背景中保持低写入，并在查询位置打开读取；在连续趋势任务中，gate 可能更平滑。如果所有位置都打开，模块近似普通状态更新；如果所有位置都关闭，模型只靠残差或局部路径。门控的价值来自输入条件与任务需求之间的对应关系，而不是门控参数的存在。

## 40.28 门控路径的可辨识性实验

要证明 gate 学到了分工，可以使用反事实而不是只看最终 loss。对同一输入保存完整前向结果，再构造三种替代：把 gate 换成训练集均值、全部置一、全部置零。定义 gate 反事实影响：

~~~math
\Delta_{\mathrm{gate}}
=L(\mathrm{do}(g=\bar g))-L(\mathrm{full}).
~~~

如果 `Delta_gate` 在长距复制、版本冲突和流式恢复任务上明显变大，说明门控至少参与了这些能力；如果所有替代的结果都接近，复杂 gate 可能没有被使用，或者残差路径已经完全承担了任务。进一步按层、时间位置、输入类型和实体事件切片，才能知道 gate 是全局规律还是个别样本的偶然。

还要检查 gate 的饱和比例和量化敏感性。门控长期接近 0/1，梯度和低精度误差的行为与连续 gate 不同；门控对微小格式变化极其敏感，则可能学到了模板捷径。报告 gate histogram、饱和率、状态范数、exact recall 和 p99，比一句“门控提高了表达能力”更有解释力。

## 40.29 状态、短卷积和 attention 的预算如何分配

混合模块的优势通常来自互补的信息路径，而不是每条路径都做完整任务。可以把每层的 token 计算和状态成本拆成：

~~~math
C_{\mathrm{layer}}
=C_{\mathrm{state}}+C_{\mathrm{conv}}
 +C_{\mathrm{attention}}+C_{\mathrm{gate}},
~~~

~~~math
M_{\mathrm{state}}
=M_{\mathrm{recurrence}}+M_{\mathrm{conv\_buffer}}
 +M_{\mathrm{KV}}.
~~~

减少 attention 层数可能降低 KV 和长上下文成本，却把更多任务交给状态压缩；增加短卷积可能改善局部混合，却增加 overlap 和 snapshot；提高 gate 维度可能增加选择性，却增加状态读写和量化难度。不能把每项成本独立相加后就宣布收益，还要测它们在同一 kernel、同一 batch 和同一设备上的交互。

一个实用的消融顺序是：纯状态、状态加 gate、状态加卷积、状态加 attention、完整混合。固定参数预算时，还要做容量匹配，否则完整模型仅仅因为参数更多而领先。固定质量时，再比较 state bytes、TTFT/TPOT、p99、恢复时间和工具/引用任务，才能判断该组合是否适合目标 workload。

## 40.30 数值稳定与长流恢复

门控状态经常包含乘法、指数衰减、归一化或 scan。长序列和低精度下，状态可能下溢、爆炸或被 gate 累积偏差放大。一个最小健康指标是：

~~~math
E_{\mathrm{restore}}
=\max_t\lVert y_t^{\mathrm{full}}
-y_t^{\mathrm{chunk+restore}}\rVert_\infty,
~~~

同时记录 `||s_t||`、NaN/Inf 比例、gate 饱和、dtype 和 chunk 长度。测试顺序应覆盖 FP32 reference、BF16/FP16 kernel、长上下文、padding、reset、batch reorder、snapshot/restore 和推测候选回滚。只验证短序列的最终字符串，无法发现状态在数万步后缓慢漂移。

服务端还要定义 state ownership。状态必须绑定 model revision、adapter、dtype、逻辑位置和请求 owner；取消或拒绝候选时要回滚临时写入；slot 复用前要清理并做 checksum。复杂混合模块的“固定状态”不是无状态，而是把历史管理从 KV allocator 转成了 state schema 和生命周期管理。

## 40.31 研究结论如何落到回退策略

Mega、gated SSM 和 H3 一类路线的公开论文可以支持其模块动机、公开公式和实验设置，但生产选择还要看具体实现、kernel、量化、长流恢复和任务边界。若门控或短卷积在目标设备上不稳定，系统应能回退到更简单的 state、attention-only 或原始 checkpoint，并记录质量和成本变化。

回退不是只切换一个模型名称。需要定义最近一次已提交状态、未提交候选状态、历史输入是否可重放、工具动作是否已产生副作用，以及用户是否需要收到降级提示。对高风险工具任务，状态不一致宁可拒绝或要求确认，也不能把一个未验证的混合状态继续交给执行器。

## 40.32 小结与资料边界

Mega 可参考 https://arxiv.org/abs/2209.10655，H3 可参考 https://arxiv.org/abs/2212.14052。本文采用概念抽象，具体结构和实验结论应查原论文。

Mega、H3 和门控状态路线说明：高效序列建模往往需要多种记忆形式协作。它们是理解今天 hybrid attention/SSM 的历史基础，而不是可以互换的同一模型。
