# Linear Attention 核分解与因果状态：研究记录

检索与核对日期：2026-11-20。

## 研究问题与范围

核查线性 attention 的有限特征分解、因果前缀递推、复杂度依赖和状态信息边界；明确该代数重排与标准 softmax attention 之间的关系，并划定第 24 章基础机制与第 25 章 Performer/FAVOR+ 近似分析的分工。

检索主题采用中英文同义扩展，包括 linear attention / 线性注意力、kernel feature map / 核特征映射、causal prefix state / 因果前缀状态、Performer / FAVOR+ / positive orthogonal random features，以及 lecture notes / derivation / 课程讲义 / 推导。查阅原始论文与出版方记录，并对照公开课程讲义的推导组织。

## 来源与核验位置

### 原始方法论文

- Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret, [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html), ICML 2020, PMLR 119:5156–5165；核对正式出版信息及出版方摘要。
- 同文 [arXiv HTML v3](https://arxiv.org/html/2006.16236)，版本日期 2020-08-31。阅读 §3.1—3.4，重点是 §3.2 的核化与复杂度、§3.3 的因果 mask/前缀状态与训练推理区别、§3.4 的递归关系；并对照 §4.1 的实验条件。浏览器逐段核对公式 (3)、(7)—(12) 及关于指数核特征维度和反向传播内存的文字。
- Krzysztof Choromanski et al., [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794v4), arXiv v4，2022-11-19。阅读摘要、§2 FAVOR+ Mechanism & Positive Orthogonal Random Features，以及相关结论。该来源支持 FAVOR+ 用正交正值随机特征估计 softmax kernel 的方法概述；随机特征误差、方差和具体算法由第 25 章继续核验。

### 教学组织参考

- David Chiang, [Chapter 9. State-Space Models and Linear Transformers](https://academicweb.nd.edu/~dchiang/teaching/tonn/2024/chapter9.pdf), CSE 60963 Theory of Neural Networks，课程讲义版本日期 2024-11-09。阅读 §9.2.1 及其相邻推导。用作从归一化 attention、特征映射到前缀汇总的教学组织参考，不作为独立的实验或普遍效果证据。
- D2L Smola, [10.6 Windowed and Linear Attention](https://d2l.smola.org/chapter_attention/windowed-linear-attention.html)。本次打开该 URL 返回 Cache miss，因此未将其作为已读依据或用于事实主张。

## 核验发现

1. 对 query/key 相似度采用非负函数并逐行归一化，可写成相似度加权的 value 和除以相似度总和。标准 scaled dot-product softmax 的相似度为指数点积 $\exp(q^{\mathsf T}k/\sqrt{d_k})$。因果情形把求和限制在 $j\le t$。
2. 若 kernel 可因子分解为有限向量内积 $K(q,k)=\phi(q)^{\mathsf T}\phi(k)$，则分子和分母都能用矩阵乘法结合律，将 query 与 key 的逐对配对改写为查询与 key 特征前缀和的乘积。这项改写精确保留所选 kernel；是否等于标准 softmax 是独立问题。
3. 按列向量书写时，因果线性 attention 的分子状态可写为 $S_t=\sum_{j\le t}v_j\phi(k_j)^{\mathsf T}$，归一化状态为 $z_t=\sum_{j\le t}\phi(k_j)$；单步分别增加当前 value 与 key 特征的外积、以及 key 特征本身。输出为 $S_t\phi(q_t)$ 除以 $\phi(q_t)^{\mathsf T}z_t$。两个状态 shape 分别为 $d_v\times r$ 和 $r$。原论文采用转置后的矩阵记法，运算关系相同。
4. 对固定特征宽度 $r$、key/query 维度 $d_k$ 和 value 维度 $d_v$，序列长度 $T$ 方向的 attention 累积成本为线性量级，核心状态读取与写入约为 $O(Trd_v)$，特征映射另有成本；Q/K/V 投影和其余 Transformer 层不由这一重排消除。单 token 解码只需要固定形状的累计状态，但这不自动限定整模型的内存。
5. 前缀和的结合性允许并行 scan/分块计算，但每个位置仍有独立前缀结果。训练反向的中间存储取决于具体实现；原论文推导特定的累计和 backward 路径，不能据此宣称所有直接递推或自动微分实现都有常数训练显存。它的实验速度数据绑定论文模型、任务、长度、硬件和实现，本章不转述为通用性能结论。
6. 原论文指出指数点积 kernel 的精确特征函数为无限维；例如 $\phi(x)=\operatorname{ELU}(x)+1$ 的正值特征是另一种可计算 kernel，不等同于精确 softmax。有限随机特征可被用于近似 softmax；Performer/FAVOR+ 的构造及估计误差属于后续专题。
7. 累计状态保存的是有限特征方向上的 key/value 汇总，而非可逐项寻址的历史 key/value 表。该表示边界本身不构成任意真实任务的容量定理；比较具体 recall 必须另有明确定义的任务与证据。

## 证据范围与未解决边界

原论文支持其数学形式、指定算法及其论文实验设置；不支持所有 feature width、模型、任务、训练实现或硬件的统一质量/性能保证。线性 attention 中“线性”须绑定 $r,d_k,d_v$ 固定这一口径，并与总模型成本分开。FAVOR+ 的无偏/近似无偏、均匀收敛和方差主张以 Performer 论文的随机特征构造与假设为准，不扩展为所有有限特征映射的性质。固定状态与独立历史条目的访问差异是状态结构的直接含义，不应替代任务级回忆实验或一般容量界。

本记录不提供本地实现、benchmark 或视觉公式渲染验证。
