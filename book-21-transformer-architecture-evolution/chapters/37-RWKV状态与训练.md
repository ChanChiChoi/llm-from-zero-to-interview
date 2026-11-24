# 第三十七章：RWKV：RNN 式状态与 Transformer 式训练

RWKV 不是一条跨版本不变的公式。RWKV-4 以逐通道衰减的加权 key-value 统计承载历史；RWKV-5/6 把每个头的状态扩成矩阵；RWKV-7 又改变了状态更新规则。本章沿这条演进线解释两种信息流：time-mixing 怎样把先前 token 的内容压进状态，channel-mixing 怎样在当前位置变换通道；整段训练为何能利用并行计算，生成时又为何只需递归传递状态。

## 37.1 RWKV 的设计目标：整段训练并行，生成时递归传状态

标准自注意力在每个位置都能同序列中已有位置的 key/value 交互。自回归生成时，这些历史表示通常逐 token 保留，因此缓存会随已处理序列变长。普通递归网络走另一条路：把过去汇总为状态，每步更新状态再处理新 token；生成状态不必按历史 token 数累加，但训练路径容易被逐步递推牵制。

RWKV 的目标是把训练和生成都放在同一套序列更新关系中：对已知训练序列，可以并行计算各位置的线性投影，并用序列扫描计算有时间依赖的 WKV；生成时，每个新 token 读取并更新前一时刻的状态。论文所说的“Transformer 式训练”指这种整段输入上的并行计算方式，不表示 RWKV 具有标准 Transformer 的 full self-attention，也不表示每个时间步彼此独立。[RWKV 原始论文](https://arxiv.org/html/2305.13048)同时给出 time-parallel 和 RNN-like inference 两种计算视图。

## 37.2 RWKV-4 的 Time-mixing 与 Channel-mixing

RWKV-4 的一个 block 交替处理时间信息和通道信息。设第 $t$ 个位置输入 block 的向量为 $x_t\in\mathbb{R}^{D}$。token shift 为不同投影准备不同的输入：每个投影把当前向量和上一位置的向量按可学习的逐通道系数混合，再映射成对应的 receptance、key 或 value。可写成 $x_t^{(z)}=\mu_z\odot x_t+(1-\mu_z)\odot x_{t-1}$，其中 $z$ 表示 $r,k,v$ 中的一种投影，$\mu_z$ 是该投影的混合系数。

Time-mixing 把 $k_t$ 和 $v_t$ 送入 WKV，得到带历史信息的向量；receptance $r_t$ 经过 sigmoid 后逐通道门控这个读出，结果再经过输出映射和残差路径。这里的 token shift 只直接混合相邻位置表示；更远历史由递归状态承载，并不是把窗口扩大到所有原文 token。

Channel-mixing 使用同样的当前/前一位置输入混合，但处理目标不同：它对 $k'_t$ 做逐通道的平方 ReLU 变换，再由 $r'_t$ 控制输出。它提供当前位置上的非线性通道变换，不独立保存一份跨时间 WKV 历史。RWKV-4 原论文以“time-mixing 管跨时间的加权信息，channel-mixing 管 token 内部的非线性变换”描述两者分工；[Johan Wind 的 RWKV 讲解](https://johanwind.github.io/2023/03/23/rwkv_details.html)也沿着这一 block 结构解释了各变量如何对应到最小实现。

## 37.3 WKV：带衰减和当前项加权的归一化状态

RWKV-4 的 WKV 可以看作逐通道归一化加权和。对当前时刻 $t$，此前位置 $i$ 的 key 决定其内容权重，距离当前越远，逐通道衰减越多；当前位置另有一个可学习的 time-first bonus $u$。每个通道上的完整关系为：

~~~math
\mathrm{wkv}_t =
\frac{
\sum_{i=1}^{t-1}\exp\bigl(-(t-1-i)w+k_i\bigr)\odot v_i
+ \exp(u+k_t)\odot v_t
}{
\sum_{i=1}^{t-1}\exp\bigl(-(t-1-i)w+k_i\bigr)
+ \exp(u+k_t)
}.
~~~

这里 $k_i,v_i,w,u$ 都是宽度为 $D$ 的向量，指数、乘法、加法和除法均按通道计算；$w$ 是非负的逐通道衰减率。分子把加权 value 累加起来，分母累加同一组权重，因此 WKV 是各通道分别归一化的历史汇总。当前位置的权重是 $\exp(u+k_t)$，其中 $u$ 使模型能特别处理当前输入；这不是把当前位置永久写入一个独立 cache。

同一运算也能写成分子状态 $a_t$、分母状态 $b_t$ 的递归：

~~~math
\begin{aligned}
a_0 &= b_0 = 0,\\
\mathrm{wkv}_t
&= \frac{a_{t-1}+\exp(u+k_t)\odot v_t}
{b_{t-1}+\exp(u+k_t)},\\
a_t &= \exp(-w)\odot a_{t-1}+\exp(k_t)\odot v_t,\\
b_t &= \exp(-w)\odot b_{t-1}+\exp(k_t).
\end{aligned}
~~~

读出当前 token 时，分子和分母状态先代表此前历史，再把带 bonus 的当前项加入这一次读出；写给后续位置的 $a_t,b_t$ 则以 $\exp(k_t)$ 更新，不把 $u$ 重复累计。这个差别正是旧稿把当前 bonus 当成线性系数乘在 key 上时会丢掉的部分。原论文附录 D 还用辅助指数 $p_t$ 对分子、分母共同重标度，避免直接计算很大的指数；因此上式说明 RWKV-4 的数学递归，并不等同于完整数值实现。[原论文 §3.1 与附录 D](https://arxiv.org/html/2305.13048)给出了该归一化式、递归形式及稳定化状态。

## 37.4 并行训练与递归推理：同一算子的两种执行方式

训练时，一条序列中的 token 都已知。各位置的 token shift 和线性投影可以批量执行；WKV 则沿时间依赖此前累计状态。RWKV-4 原论文把 WKV 称为按时间推进的 scan，并指出主要矩阵乘法可并行。递归更新可表达为一系列仿射状态变换，因而可使用 scan、分块扫描或专用 kernel 加速；这些执行方法保留了 token 顺序依赖，并没有把位置之间的关系删掉。

自回归生成时，下一 token 依赖当前模型输出，所以模型仍需按 token 推进。每层只需带上这一版本所需的 shift 输入和 WKV 状态，处理新 token 后写回下一状态。整段训练与逐 token 生成使用的是同一因果更新关系，只是输入已知程度和执行组织不同。RWKV-5/6/7 的论文也给出了每头矩阵状态的递归与并行形式；具体并行效率取决于算法和实现，不能从“可并行”直接推出某硬件上的吞吐优势。[Eagle and Finch](https://openreview.net/pdf?id=soz1SEiPeq)与[RWKV-7 论文](https://openreview.net/pdf/74011cffc1082ac9394afde92802c810c51baabf.pdf)分别展示了后续版本的状态更新和并行表示。

## 37.5 RWKV-5/6/7：从矩阵状态到可修正记忆

RWKV-4 的 numerator/denominator 是逐通道向量状态。RWKV 后续版本改变了每个头承载的状态对象和更新能力；这些版本不能用 RWKV-4 的归一化公式替代。

| 版本 | 状态与主要更新 | 相对前版的变化 |
| --- | --- | --- |
| RWKV-4 | 逐通道 numerator/denominator 向量，按固定的可学习衰减累计，并在读出中加入当前项 bonus。 | 以紧凑的加权统计替代逐位置保存整段 key/value。 |
| RWKV-5 Eagle | 每个 head 保存矩阵形状的 key-value 状态；状态按可学习的逐通道衰减更新，并配合 head 级归一化与门控。 | 从向量统计转为矩阵状态，能在每个头中表示更多 key-value 关联。 |
| RWKV-6 Finch | 保留每头矩阵状态；token shift 和衰减改为依赖输入的动态形式。 | 当前和前一位置的表示可调节输入混合与每通道衰减，而不再只使用固定的 shift/decay。 |
| RWKV-7 Goose | 每头矩阵状态经过逐通道衰减和方向性移除，再写入新的 value-key 关联。 | 以广义 delta 规则更新记忆；论文把 v6 的输入依赖插值改回较简单的 token shift，将主要变化放在状态演化。 |

在 RWKV-5/6/7 中，矩阵状态表示每个头所维护的 key-value 关联；把它称为“更大的向量状态”会掩盖其二维读写结构。[Eagle and Finch 论文](https://openreview.net/pdf?id=soz1SEiPeq)报告 RWKV-5 的矩阵状态和固定 decay，以及 RWKV-6 的输入依赖 token shift 与动态 decay。RWKV-7 仍按每头矩阵状态运行，但改用可修正的状态演化。对单个 head，设状态为 $S_t\in\mathbb{R}^{d_h\times d_h}$，其递归可写为：

~~~math
S_0=0,\qquad
S_t=S_{t-1}\left(\mathrm{diag}(w_t)
-\hat{\kappa}_t^{\mathsf T}(a_t^{\mathrm{iclr}}\odot\hat{\kappa}_t)\right)
+v_t^{\mathsf T}\tilde{k}_t.
~~~

其中 $w_t$ 是逐通道衰减，$\hat{\kappa}_t$ 是归一化的 removal key，$a_t^{\mathrm{iclr}}$ 是逐通道的 in-context learning rate，$\tilde{k}_t$ 是 replacement key，$v_t$ 是待写入的 value。这里用上标将 ICL rate 与 RWKV-4 的 numerator 状态 $a_t$ 区分开；RWKV-7 论文将 ICL rate 记为 $a_t$。括号中的对角衰减压低旧状态；后面的秩一减项沿 removal key 方向修正旧关联；最后的外积把新关联写入状态。移除方向与写入 key 被分别定义，因此更新不只是“把所有旧记忆一起衰减，再把新 key-value 相加”。更新后的每头状态由 receptance 读出，再经过 head normalization、当前项 bonus、gate 和输出映射；具体组合随版本而变。RWKV-7 的形式化状态追踪结论限于论文给出的理论设定；本节只把它用于解释更新机制，不将其扩展为通用语言模型优越性主张。[RWKV-7 正式论文](https://openreview.net/pdf/74011cffc1082ac9394afde92802c810c51baabf.pdf)给出 removal/replacement key 定义、状态递归和理论边界。

## 37.6 固定状态的含义、版本证据与 RWKV-8 边界

“状态不随上下文长度增长”只表示处理更多 token 时，不必为每个历史位置继续追加一份状态。它不表示状态大小与模型宽度、层数或头维度无关。RWKV-4 原论文附录 D 描述每层五个宽度为 $D$ 的向量状态：time-mixing 输入、channel-mixing 输入、WKV 分子、WKV 分母和稳定化指数；按该表述，状态元素数随层数约为 $5LD$。RWKV-5/6/7 的每头矩阵状态则按层和 head 维度增长，量级为 $Lh d_h^2$，还需加上各版本的 shift 等状态。因此“固定”是相对于已读 token 数 $T$ 而言，并非小到可以忽略模型维度。

这些递归状态是模型学到的压缩表示，不是可按 token 位置任意寻址的原文副本。由此可以说明它与显式历史 key/value 的访问方式不同；但状态维度本身不能证明某类精确召回任务必然失败，也不能证明模型已具备可靠的任意检索能力。检索和复制结论需要固定模型版本、任务构造与评测条件；跨架构选型由第四十二章承接，外部检索结构由第四十五章承接。

截至 2026-11-23，RWKV 官方 wiki 将 RWKV-8/Heron 标为实验阶段，并列出 DeepEmbed 与 ROSA。官方材料把 DeepEmbed 描述为按 token index 读取、参与 FFN 的 learned embedding；ROSA 则是后缀自动机式的序列检索组件。它们与 v4—v7 的 WKV 状态方程属于不同组件层次；本次查阅到的官方材料和作者页面未呈现一篇统一的同行评审 RWKV-8 论文来替代前述状态更新链。因而这里把它们作为在研扩展标记，不据此推出成熟度、普遍性能或 RWKV-4—7 的统一新公式。[RWKV 官方架构资料](https://www.rwkv.cn/docs/RWKV-Wiki/RWKV-Architecture)列出 RWKV-8 的实验状态、DeepEmbed 与 ROSA；[作者的 RWKV-8 页面](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md)则将其放在 “RWKV-8 and beyond” 的探索框架中。
