# Gated DeltaNet：递归更新、门控职责与 chunkwise 证据

研究日期：2026-11-24  
研究范围：核对 Gated DeltaNet 的状态方向、delta correction 与 scalar decay 的关系，理解在线回归解释和 chunkwise WY 训练；复核与 KDA、2026 年 Gated DeltaNet-2 的公式边界。这里的研究不覆盖 serving runtime、缓存生命周期或生产部署经验。

## 检索记录

- 平台：arXiv HTML/摘要、GitHub、作者技术博客与公开课程式笔记。
- 检索词：`Gated Delta Networks arXiv 2412.06464 equation 10 chunkwise`；`Gated DeltaNet-2 arXiv 2605.22791`；`Gated DeltaNet explained tutorial recurrence chunkwise WY blog`；`Gated Delta rule online regression memory state tutorial`；`Gated DeltaNet chunkwise WY lecture notes tutorial`；`site:github.com Gated DeltaNet mathematical reference chunkwise recurrence`。
- 选择依据：以原论文核对更新方程、实验条件和算法边界；并行查找 delta rule 与 chunkwise WY 的读者向推导，用于确认适合解释的关系；不从搜索摘要推断模型普遍效果或部署行为。

## 主要研究发现

### Gated DeltaNet 的 canonical 状态更新

Songlin Yang、Jan Kautz、Ali Hatamizadeh，《Gated Delta Networks: Improving Mamba2 with Delta Rule》，arXiv:2412.06464v3，2025-03-06。原文位置：§2.2 Delta Networks、§3.1 Eq. 10、Table 1、§3.2 S-NIAH、§3.3 Chunkwise Training、§3.4 Token Mixer/Hybrid Models、§4 Setup/Throughput。

- 来源：[arXiv v3 摘要](https://arxiv.org/abs/2412.06464v3)、[固定版本 HTML](https://arxiv.org/html/2412.06464v3)。
- 原论文使用 $S_t\in\mathbb{R}^{d_v\times d_k}$，更新式为 $S_t=S_{t-1}(\alpha_t(I-\beta_t k_tk_t^\top))+\beta_t v_tk_t^\top$。$\alpha_t$ 是控制旧状态衰减的数据相关标量；$\beta_t$ 是 delta 更新强度。论文将状态视为 key-to-value 的 fast-weight，并把 delta correction 联系到平方回归损失的一步 SGD。
- 统一使用 $A_t=S_t^\top\in\mathbb{R}^{d_k\times d_v}$ 后，等价更新为 $A_t=\alpha_t(I-\beta_tk_tk_t^\top)A_{t-1}+\beta_tk_tv_t^\top=\alpha_tA_{t-1}+\beta_tk_t(v_t-\alpha_tA_{t-1}^\top k_t)^\top$。后一式明确了先衰减旧状态，再从衰减状态得到预测残差并沿当前 key 写入。
- 原文 §3.3 在 DeltaNet 的 WY chunk 算法上纳入累计衰减，用块内批量线性代数处理 token 交互，并将块边界状态继续传递。训练排布改变不改变逐 token 递归语义；论文没有据此证明所有 kernel、硬件和负载都获得同样 wall-clock 收益。
- §3.4 所示 Gated DeltaNet 模块还包括 query/key/value 投影、短卷积、SiLU、Q/K L2 normalization、由投影产生的 $\alpha/\beta$ 和独立 output gate。模块中的 output gate 不属于 Eq. 10 的状态衰减变量。
- §4 的主要匹配训练设置包括 1.3B 参数、FineWeb-Edu 100B token、4K training sequence；混合模型采用 2K sliding window。吞吐图是单张 H100 的训练测量。它们是作者报告的特定实验，不支持生产模型、任意长度或所有任务上的普遍性能结论。

### Gated DeltaNet-2 对 erase/write 的扩展

Ali Hatamizadeh、Yejin Choi、Jan Kautz，《Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention》，arXiv:2605.22791v2，版本页标注 2026-11-06。原文位置：§2.2 Gated DeltaNet/KDA 更新、§3.1 Eq. 8–10、§3.2 Fast-weight View、§3.3 Chunkwise Training、§4 实验、Appendix C–E。

- 来源：[arXiv 摘要](https://arxiv.org/abs/2605.22791)、[v2 固定 HTML](https://arxiv.org/html/2605.22791v2)、[作者官方实现仓库](https://github.com/NVlabs/GatedDeltaNet-2)。
- 论文在 $A\in\mathbb{R}^{d_k\times d_v}$ 方向下定义 $\bar A_t=D_tA_{t-1}$、$e_t=b_t\odot k_t$、$z_t=w_t\odot v_t$，并更新 $A_t=\bar A_t+k_t(z_t-\bar A_t^\top e_t)^\top$。$b_t$ 逐 key 通道选择 erase/read 方向，$w_t$ 逐 value 通道选择写入；channel-wise decay $D_t$ 仍负责旧状态保留。
- 按论文参数约束令 $b_t$ 与 $w_t$ 收缩到同一 scalar gate 可得到 KDA；再将 $D_t$ 收缩成 scalar identity 才得到 Gated DeltaNet。此为公式层级的特例关系，不能仅凭模型名称推断实际模型采用该参数化。
- GDN-2 论文的主要匹配设置也以 1.3B 参数、100B FineWeb-Edu token、4K 训练长度为基线；部分吞吐测量为单张 H100 的 hybrid 模型。检索任务、模型配置、速度和消融结果均为作者报告，尚不等于独立复现或生产经验。

### 面向读者的 delta rule 与 WY 解释

- Erik Steiger，《Chunkwise linear attention and DeltaNet (Part 3)》，2026-05-16，§2 Chunkwise、§3 WY、DeltaNet 与 §5 Implementation。来源：[文章](https://www.ersteiger.com/posts/linear-attention-part-3/)。将每步 delta transition 写成 identity 减 rank-one 项，再解释块内乘积的 WY 表示 $I-W^\top Y$；强调避免显式构造 $d_k\times d_k$ 的连乘，并指出块内主要转成矩阵运算、块间继续传递状态。文章中的 microbenchmark 和 MQAR 数字只支持其自述实现与设置。
- Brandon Behring，课程笔记《Delta-rule lineage: DeltaNet, Gated DeltaNet, Kimi Linear》，章节 12.2、12.5、12.6。来源：[课程笔记](https://ssm-foundations.brandon-behring.dev/chapters/ch12-delta-rule-lineage/)。以平方召回误差的一步梯度更新解释 delta rule 的两种等价代数形式，再将其与全局 gated forgetting 并列，适合作为 online-learning 和 erase/rewrite 讲解入口。该页面及配套实验是作者教学材料，不能替代原论文对算法、实验或一般稳定性的定义。

## 书稿使用边界

- 正文选用 $A=S^\top$ 的 key-by-value 方向，并在章节内固定公式、读出与小型状态轨迹的形状；引用原论文时说明原论文采用转置方向，避免矩阵乘法左右混淆。
- 把 $\alpha_t$ 解释为全局旧状态衰减，把 $\beta_t$ 解释为当前 key 上残差写入强度。两者既不等价，也不应与模块末端 output gate 混为一谈。
- 通过论文所用匹配训练条件限定实验主张；不保留原草稿中无来源的通用 gate histogram、scheduler、speculative rollback、租户安全、cache manifest 或生产 kernel 清单。
- KDA 的逐 key 通道衰减细节由第 61 章承担；GDN-2 仅作为有固定版本来源的后续公式边界；混合注意力的整体路径比较交由第 65 章。
- 未进行本地算法复现、kernel benchmark 或用户试读；所列数值/方法结论为 source-backed，不是 local-demo。
