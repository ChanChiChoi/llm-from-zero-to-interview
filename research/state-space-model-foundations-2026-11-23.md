# 状态空间模型基础：连续系统、离散化与三种计算表示

研究日期：2026-11-23。用途：第二十一册第 29 章《状态空间模型 SSM 的数学直觉》的职责确认、公式核验与教学路径设计。编辑决定和正文修改见 `progress/book-21-transformer-architecture-evolution/chapter-29-audit.md`；第 30 章负责跨架构信息流比较，第 31 章负责 S4/HiPPO 深入机制，第 32 章负责 Mamba 的输入选择性与 selective scan。

## 研究问题与检索

核心问题是：如何在不把状态空间模型等同于某一种实现的前提下，讲清连续时间线性系统如何变成 token 递归、递归为何能展开为因果卷积，以及时间尺度和稳定性各自说明什么。

检索词包括 `structured state space sequence model continuous discrete recurrence convolution`、`HiPPO online compression polynomial projection`、`SSM zero-order hold discretization recurrence impulse response` 和 `state space model lecture notes convolution recurrence`。先读 S4、HiPPO 原论文的定义与相关小节，再用教材和课程讲义交叉检查符号、离散化假设和概念顺序；没有把论文实验速度、长序列基准分数或当前模型采用情况写成普遍结论。

## 核心发现

1. **连续方程定义的是局部动力学。** 在线性系统中，用 $A,B,C,D$ 作为四个矩阵的记号，依次描述状态演化、输入注入、状态读出和直接输入通路。它是序列层的一种数学表示；不应把一般 SSM、RNN 和具体 S4/Mamba block 当作同义词。
2. **离散化必须带假设。** 若每个采样区间内输入保持常数（zero-order hold，ZOH），对连续线性方程逐步精确积分。状态转移项为 $\bar{A}=\exp(\Delta A)$ 这一矩阵指数；输入写入项为 $\bar{B}=\int_0^\Delta\exp(\tau A)B\,\mathrm{d}\tau$ 这一积分。显式 Euler 的一阶离散参数可写成 $(\bar{A},\bar{B})\approx(I+\Delta A,\Delta B)$ 这一组近似。S4 原论文使用双线性变换，故不能将 ZOH 或 Euler 写成所有 SSM 实现的共同选择。采样步长与连续参数共同决定每步状态转移。
3. **递归和卷积是固定线性时不变系统的两种计算视图。** 当 $(\bar{A},\bar{B},C)$ 固定且初始状态为零时，递归展开得到第 $j$ 个冲激响应 $K_j=C\bar{A}^{j}\bar{B}$ 这一核项，它表示输入经过 $j$ 步状态转移后对输出的影响。输出是历史输入与该因果冲激响应的卷积，并含有直接项 $Du_t$ 这一项。将直接项并入核时，零延迟核可写为 $D+K_0$ 这个合并值。卷积表示为整段并行提供了算法入口，但核的生成、应用方式和实现成本仍需单独考虑；原论文明确指出高效计算核本身是 S4 的技术重点。
4. **时间常数刻画单个线性模态的衰减，不等于任务记忆。** 标量递推可写为 $x_t=ax_{t-1}$ 这一形式，令 $a$ 表示每步衰减系数且取值位于零与一之间。过去输入经过若干步后的影响按 $a^k$ 缩放。半衰期公式给出衰减到一半所需的步数；它只描述该线性模态直接影响的衰减，不保证事实可分、可寻址、可读出或在任务上可恢复。矩阵系统需考虑完整谱和更新结构，不能把单标量结果泛化为模型能力结论。
5. **边界应停在基础机制。** HiPPO 研究的是按指定时间测度把历史函数在线投影到多项式基上的近似问题；它不表示存下每个 token 的显式地址。S4 对 HiPPO 相关矩阵进行结构化处理，细节归第 31 章。Mamba 使部分状态更新参数依赖输入，固定 LTI 卷积核的解释不再直接适用于所有选择性更新；其计算路径归第 32 章。第 29 章只提示这些边界，不承担后续章节的推导。

## 来源与使用范围

| 来源 | 阅读位置 | 支持内容 | 限制与本章用法 |
| --- | --- | --- | --- |
| Albert Gu、Karan Goel、Christopher Ré，[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)，arXiv:2111.00396，发表于 ICLR 2022 | §2.1 连续时间 latent state、§2.3 离散递归、§2.4 卷积表示 | 连续 SSM 定义、离散转移、固定初始状态下递归展开为因果卷积；论文采用双线性离散化，且指出高效生成卷积核是其方法重点 | 作为 S4 论文的模型定义与方法证据。论文省略 $D$ 仅为 exposition；不把其复杂度或 benchmark 数字推广到任意状态维数、实现和硬件。 |
| Albert Gu、Tri Dao、Stefano Ermon、Atri Rudra、Christopher Ré，[HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669)，arXiv:2008.07669，NeurIPS 2020 | 摘要、§1、§2 及连续到离散讨论 | HiPPO 的目标是按给定 measure 在线压缩连续信号/离散序列并投影至多项式基；其“最优”相对于论文定义的函数近似问题 | 仅用于交代 HiPPO 的历史/概念边界和指向第 31 章；不把它说成无损保存任意 token，也不采用其任务精度作普遍效果证据。 |
| [Dive into Deep Learning：12.4 State Space Models](https://d2l.smola.org/chapter_recurrent-modern/ssm.html)，网页教材，访问日期 2026-11-23 | §12.4.1.1–12.4.1.4 | 将连续动力学、ZOH、时间尺度、卷积核和三种等价视图排成基础教学路径；给出标量 ZOH 和半衰期推导 | 用作解释顺序与推导交叉检查，页面未显示固定 revision；不替代 S4/HiPPO 原论文，也不把其中的模型或 benchmark 结果作为本章结论。 |
| David Chiang，[Theory of Neural Networks, Chapter 9: State-Space Models and Linear Transformers](https://academicweb.nd.edu/~dchiang/teaching/tonn/2024/chapter9.pdf)，课程讲义，页脚版本日期 2024-11-09 | §9.1.2–9.1.3 | 用 forward Euler 推导离散递归，并由递归替换推导因果卷积；明确不同离散化会产生不同 `Ā`、`B̄` | 用作第二个教学来源核查 Euler 与卷积推理；不作论文实验或工程性能的证据。 |

会议版本和出版信息另以 [ICLR 2022 OpenReview 论文版](https://openreview.net/attachment?id=uYLFoz1vlAC&name=pdf)核对 S4，以 [NeurIPS 2020 论文页](https://proceedings.neurips.cc/paper_files/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)核对 HiPPO；公式与方法细节仍以表中 arXiv 原文对应章节为主要阅读入口。

## 证据状态与未核项

- 连续方程、ZOH 条件下的精确积分、Euler 近似、递归展开和卷积核关系：`source-backed`；采用已列论文与教材公式，并对索引与当前输入项作了符号口径统一。
- `半衰期`公式：由标量递推 `a^k` 与定义 `a^k=1/2` 推得，只表示单个稳定标量模式的线性衰减。
- 固定 LTI 系统的卷积表述不意味着卷积路径总比递归快，也不意味着一般 SSM 都是 LTI。dense 状态矩阵的每步代价与对角/结构化矩阵不同；本章不给脱离维数和实现的统一 wall-clock 结论。
- 未实现、运行或 benchmark 任何模型、代码、kernel；未核查具体框架中 SSM 的参数初始化和部署状态契约，这些不属于本章读者结果。
