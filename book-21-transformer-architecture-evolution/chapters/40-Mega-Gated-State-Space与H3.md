# 第四十章：Mega、Gated State Space、H3 等早期混合路线

## 40.1 语言建模缺口如何引出早期混合设计

状态空间模型可以用卷积并行处理整段输入，也能以递归状态逐 token 更新；但“能传播很远”不等于“能按当前内容比较远处 token”。H3 论文把这个差异放到两类小型合成语言任务中检查。

Induction Head 任务要求模型记住特定标记后出现的 token，并在后续标记处复现它。Associative Recall 则把若干 key-value 对写进上下文，再在末尾给出一个 key，要求模型取回对应 value。它们将“记住事件后的信息”和“比较当前位置与历史内容”分开观察，能帮助定位架构的能力缺口；它们仍是受控合成任务，不能替代自然语言 ICL 或真实文档检索评估。[H3 原论文 §3.1](https://arxiv.org/abs/2212.14052v3)

Mega、GSS 和 H3 对信息流作了不同安排。Mega 将时间平滑引入 gated attention；GSS 用状态空间层形成上下文，再用另一条输入流调节结果；H3 则把前序线索、跨位置关联和查询读出组织成两级 SSM。理解它们的差异，要跟踪每条信息经过哪些运算，而不是只看模型名称里有没有 gate 或 state。

## 40.2 Mega：多维 EMA 怎样进入 gated attention

Mega 的移动平均模块不只是输出一条独立的平滑分支。它为每个输入通道维护一个扩展状态，并让该状态以可学习的阻尼系数递归吸收当前输入。对通道 $j$，论文的多维 damped EMA 可写为：

~~~math
\mathbf{u}^{(j)}_t=\boldsymbol{\beta}_j x_{t,j},
\qquad
\mathbf{h}^{(j)}_t=
\boldsymbol{\alpha}_j\odot\mathbf{u}^{(j)}_t+
(1-\boldsymbol{\alpha}_j\odot\boldsymbol{\delta}_j)\odot
\mathbf{h}^{(j)}_{t-1},
\qquad
x'_{t,j}=\boldsymbol{\eta}_j^{\mathsf T}\mathbf{h}^{(j)}_t.
~~~

$\mathbf{u}^{(j)}_t$ 将一个输入维度展开到多个 EMA 维度；$\mathbf{h}^{(j)}_t$ 以不同衰减尺度积累历史，再由 $\boldsymbol{\eta}_j$ 投影回原维度。阻尼项让旧状态的保留比例不必与当前输入的写入比例互为补数，因此它比单一固定系数的指数平均更灵活。

得到整段上下文表示 $\mathbf{X}'=\mathrm{EMA}(\mathbf{X})$ 后，Mega 用它形成共享特征 $\mathbf{Z}$，再据此构造 query 和 key；value 则由原始输入 $\mathbf{X}$ 投影得到。attention 仍计算位置间的内容分数：

~~~math
\mathbf{O}
=f\left(\frac{\mathbf{Q}\mathbf{K}^{\mathsf T}}{\tau(\mathbf{X})}
+\mathbf{b}_{\mathrm{rel}}\right)\mathbf{V}.
~~~

其中 $f$ 可以是 softmax 等 attention 函数，$\mathbf{b}_{\mathrm{rel}}$ 是相对位置项。论文还提出了 Laplace attention function，以误差函数构造有界的注意力函数，缓解平方 ReLU 函数的值域和梯度无界问题；这是 attention 函数的选择，与 EMA 提供的时间结构偏置承担不同作用。[Mega 原论文 §3.2](https://arxiv.org/abs/2209.10655v3)

EMA 为 query/key 提供平滑后的局部时间上下文；attention 仍根据 query-key 内容关系汇集 value。接下来，Mega 从 EMA 上下文 $\mathbf{X}'$ 计算 reset gate 和 update gate。reset gate 调制 attention 输出，候选表征将这部分结果与 EMA 上下文合并；update gate 再在候选表征与原始输入 $\mathbf{X}$ 之间插值：

~~~math
\begin{aligned}
\boldsymbol{\gamma}
&=\mathrm{SiLU}(\mathbf{X}'W_{\gamma}+b_{\gamma}),\\
\boldsymbol{\varphi}
&=\mathrm{sigmoid}(\mathbf{X}'W_{\varphi}+b_{\varphi}),\\
\widehat{\mathbf{H}}
&=\mathrm{SiLU}\left(
\mathbf{X}'W_h+(\boldsymbol{\gamma}\odot\mathbf{O})U_h+b_h
\right),\\
\mathbf{Y}
&=\boldsymbol{\varphi}\odot\widehat{\mathbf{H}}
+(1-\boldsymbol{\varphi})\odot\mathbf{X}.
\end{aligned}
~~~

所以完整 Mega 保留全长 attention，其 attention 计算仍随序列长度呈二次增长。论文的 Mega-chunk 把 attention 限制在长度为 $c$ 的块内，attention 项约为 $O(Lc)$；当 $c$ 固定时，这部分随长度 $L$ 线性增长。整段 EMA 仍把较早上下文带入块内表示，但这种压缩后的跨块影响不等于任意块之间都能直接做 query-key 配对。[Mega 原论文 §§3.1—3.5](https://arxiv.org/abs/2209.10655v3)

## 40.3 GSS：以门控缩小状态空间上下文路径

Gated State Space（GSS）借鉴 Gated Attention Unit 的两路结构，把 GAU 中二次复杂度的 attention 换成对较窄表示执行的 DSS。输入 $\mathbf{X}$ 先生成状态路径 $\mathbf{U}$ 和门控路径 $\mathbf{V}$，DSS 只处理 $\mathbf{U}$；结果映射到 $\mathbf{V}$ 的宽度后逐点相乘，再投影回模型维度：

~~~math
\mathbf{U}=\mathrm{GELU}(\mathbf{X}W_1),
\qquad
\mathbf{V}=\mathrm{GELU}(\mathbf{X}W_2),
\qquad
\mathbf{Y}=\mathrm{DSS}(\mathbf{U}),
\qquad
\mathbf{U}_{\mathrm{context}}=\mathbf{Y}W_3,
\qquad
\mathbf{O}=(\mathbf{U}_{\mathrm{context}}\odot\mathbf{V})W_4.
~~~

可沿两路输入理解这个 block：DSS 把较窄的 $\mathbf{U}$ 沿序列上下文化，$\mathbf{V}$ 在相同位置控制哪些上下文特征进入输出。论文的常用配置令 DSS 宽度 $H=E/4$，其中 $E$ 是模型维度；作者指出 TPU 上的 FFT 运算是训练瓶颈，缩窄这条路径能减少它的工作量。门控发生在两条投影流的乘法处；DSS 的状态参数仍是共享的线性结构，不是由每个 token 动态生成的状态转移矩阵。[GSS 原论文 §3.2](https://arxiv.org/abs/2206.13947v3)

GSS 本身可以作为 attention 的替代层。论文还测试了 GSS-Hybrid：在多数 GSS 层之间按固定间隔插入标准 Transformer block。这个变体保留了显式 attention 路径；它说明“混合”也可以发生在层与层之间，而不只发生在一个 block 内。

## 40.4 H3：shift 与 diagonal SSM 怎样组合 key-value 记忆

H3 针对前面两种合成任务设计两类状态路径。shift SSM 把前序 token 的信息按位置向后传递；diagonal SSM 则累积长程状态。模型把输入投影为 $\mathbf{Q},\mathbf{K},\mathbf{V}$，先让 shift SSM 处理 key 路径，再与 value 做逐点交互，最后让 diagonal SSM 保存这些交互并由 query 读取。单一维度的简化形式是：

~~~math
\mathbf{o}_t=
\mathbf{q}_t\odot
\left[
\mathrm{SSM}_{\mathrm{diag}}
\left(
\mathrm{SSM}_{\mathrm{shift}}(\mathbf{k})\odot\mathbf{v}
\right)
\right]_t.
~~~

用序列「a 2 c 4 b 3 d 1 a」看这条路径：在 token 2 处，shift 路径能把前一位置的 key a 与当前位置的 value 2 对齐；乘法交互把这对信息写入后续状态。读到末尾的 query a 时，query 路径与累积关联比较，从状态中取出与 a 对应的 value。这个文本示意只解释对齐关系；真实模型通过学习到的向量表示完成运算。

在多头形式中，模型把每个 head 的 key 和 value 做外积，形成矩阵状态，再用 query 读取：

~~~math
\overline{\mathbf{K}}^{(h)}
=\mathrm{SSM}_{\mathrm{shift}}(\mathbf{K}^{(h)}),
\qquad
\mathbf{R}^{(h)}_t
=\mathrm{SSM}_{\mathrm{diag}}
\left(
\overline{\mathbf{K}}^{(h)}_t
(\mathbf{V}^{(h)}_t)^{\mathsf T}
\right),
\qquad
\mathbf{O}^{(h)}_t
=\mathbf{Q}^{(h)}_t\mathbf{R}^{(h)}_t.
~~~

shift SSM 提供对前序位置的延迟记忆，乘法使当前内容能够与它交互，diagonal SSM 将这些交互带过更长历史。H3 block 按 Transformer 风格与 MLP、残差连接和层归一化组合。这里的内容选择来自投影之间的乘法；状态转移本身不是按 token 动态改写的。[H3 原论文 §§3.1—3.2](https://arxiv.org/abs/2212.14052v3)

## 40.5 三条路线的混合位置与 Hyena 谱系

| 路线 | 序列上下文怎样形成 | 混合发生的位置 | attention 的角色 |
|---|---|---|---|
| Mega | 多维 EMA 平滑历史，attention 按 query-key 分数汇集 value | EMA 表示进入 gated attention 的特征与更新路径 | 完整 Mega 保留全长 attention；Mega-chunk 限制在块内 |
| GSS | DSS 上下文化较窄的投影流，再与另一投影逐点相乘 | 同一层的状态路径与门控路径交互 | GSS 核心层用 DSS 替换 GAU attention；GSS-Hybrid 可插入 Transformer block |
| H3 | shift SSM 对齐前序 token，diagonal SSM 积累 key-value 交互供 query 读取 | 同一层内两类 SSM 与乘法交互组合 | H3-attention 在 H3 block 之间保留少量完整 attention 层 |

三种方案都含门控或乘法操作，但计算路径不同：Mega 仍显式构造 query-key attention；GSS 让 gate 调节 DSS 产生的上下文；H3 把输入比较嵌进状态交互。Hyena 原论文把 GSS 和 H3 分别视作特定滤波器参数化下的低阶 Hyena 算子；这是算子层面的包含关系，完整模型层数、block 组成、训练设置和实验结果仍需按各自论文判断。[Hyena Hierarchy 原论文](https://proceedings.mlr.press/v202/poli23a.html)

本章聚焦这些早期结构如何分配信息路径。Hyena 的隐式长卷积与阶递归已在第三十九章展开；卷积家族的归纳偏置和硬件实现由第四十一章承担；面向任务的架构取舍留给第四十二章。

## 40.6 原论文结果及其适用条件

三篇论文的结果来自不同任务、模型和实现，适合用来说明每种设计在什么问题上取得了什么结果。

| 路线与测量 | 论文设置 | 结果和边界 |
|---|---|---|
| Mega：Long Range Arena | 六项任务，输入长度约 $1$K—$16$K；Mega 的模型规模按任务调整到接近 S4-v1；结果取 $5$ 个随机种子平均 | Mega 平均准确率为 $88.21$，Mega-chunk 为 $85.66$；Mega-chunk 的块长通常为 $128$，Path-X 为 $4096$。表中 S4-v1 为 $80.48$，S4-v2 为 $86.09$，但作者说明 S4-v2 用了更大的模型和更好的调参。结果限于该 benchmark 与配置。 |
| GSS：PG19 语言建模与吞吐 | 固定参数表中，GSS 约 $192$M、DSS 约 $209$M；训练长度 $4$K，非基线模型使用 $64$ 个 TPUv4 cores、$125$K steps | 在 $4$K 评测长度下，GSS 报告 PPL $12.84$、吞吐 $5.3$ steps/s；DSS 为 $13.53$、$1.8$ steps/s。GSS 的状态初始化在 PG19 上重调，DSS 使用其默认初始化，所以长度外推不是完全匹配的初始化对照。 |
| H3：合成语言能力 | 两层模型；Induction Head 长度 $30$、词表 $20$；Associative Recall 长度 $20$、词表 $10$ | 按两项任务顺序，H3 准确率为 $100.0/99.8$，S4D 为 $35.6/86.0$，GSS 为 $6.8/78.0$，attention 为 $100.0/100.0$。这支持 H3 在这两种合成任务上的设计动机，不等于证明一般 ICL 或自然语言检索。 |
| H3-attention：Pile 语言建模 | $125$M Hybrid H3 在 The Pile 上训练 $400$B tokens；与公开 GPT-Neo-125M checkpoint 比较，使用相同 GPT-2 tokenizer | 论文报告 Pile PPL 为 $8.8$，GPT-Neo-125M 为 $9.4$。两边数据集相同，但 checkpoint 与训练 recipe 并非完全匹配；这个结果不能单独识别 H3 层的因果贡献。 |
| H3-attention：生成吞吐 | $1.3$B 模型、A100 80GB、batch $64$，prompt 长度 $512/1024/1536$，每条序列生成 $128$ tokens | 论文报告最高约 $2.4\times$ 吞吐；这是指定模型、kernel 和负载下的系统测量，不能视作所有 H3 block 或硬件上的固定加速。 |

Mega 的原论文还比较了多种数据模态；GSS 与 H3 论文也包含更多训练和系统实验。这里保留的数字只用于呈现设计证据的类型与边界。判断某种路线是否适合新任务，仍需对齐目标数据、模型规模、训练预算和执行实现。[Mega 原论文](https://arxiv.org/abs/2209.10655v3)；[GSS 原论文](https://arxiv.org/abs/2206.13947v3)；[H3 原论文](https://arxiv.org/abs/2212.14052v3)
