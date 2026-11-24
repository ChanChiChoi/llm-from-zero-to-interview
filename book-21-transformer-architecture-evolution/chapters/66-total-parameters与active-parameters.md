# 第 66 章 Total Parameters 与 Active Parameters：按统计口径读懂模型规模

MoE 模型发布时常同时写着“总参数很多”和“每个 token 只激活一部分”。这两个数字不矛盾：一个数模型中纳入统计的全部参数，另一个数按特定口径统计的单 token 路径。要读懂它们，先把模型里有哪些权重、一个 token 经过哪些专家、一个请求触及哪些专家，以及这些权重占多少字节分开。

本章只讲参数数字的含义和比较边界。MoE 与 SSM、Attention 的组件分工见[第 47 章](47-MoE-SSM-Attention组合.md)，Mixtral 的架构案例见[第 55 章](55-Mistral-Mixtral高效与MoE.md)；训练 FLOPs 与预算计算见[第五册第 13 章](../../book-05-llm-training/chapters/13-训练成本与资源规划.md)。

## 66.1 两个数字回答不同问题

看到“模型有多少参数”，先问这是什么统计口径。Total Parameters 和 Active Parameters 关注的对象不同；它们也都不能直接替代存储、计算或服务测量。

### 66.1.1 Total Parameters：清点模型的参数库存

Total Parameters（总参数量）通常统计报告口径纳入的全部模型参数：共享层、所有专家，以及其他被计入的参数。一个 MoE 层即使有很多专家，对于总参数量来说，这些专家仍都属于模型的参数集合。

统计时要看来源是否包含 embedding、output head、router、shared expert，是否把 tied weights（共享权重）重复计数，以及是否单列 non-embedding parameters。模型卡若同时报告 total 与 non-embedding 两个字段，说明“模型参数总数”本身也需要记录定义，不能仅靠模型名字猜。

### 66.1.2 Active Parameters：追踪一个 token 经过的权重路径

Active Parameters（激活参数量）通常描述一个 token 在前向路径中会使用多少参数。MoE 的 router 在每层根据当前 token 的隐藏表示选出若干专家；该 token 经过被选中专家，同时经过按报告口径计入的共享路径。下一层会重新路由，因此一个 token 并没有一组贯穿全模型不变的专家。

课程讲义常用等大小专家说明这一点：若每个专家有 $P$ 个参数，一层有 $E$ 个专家，而 token 选中 $k$ 个，那么专家库共有 $EP$ 个参数，这个 token 选中的专家部分是 $kP$。讲义还指出，router 与通信有额外成本，保存模型时仍要考虑整组专家参数。[Stanford MS&E 319 课程讲义 §1.1](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf)

这里的 active 是一种按 token 路径统计的参数口径，并非严格的 FLOPs 计数。来源是否把注意力、router、共享层、embedding 等一并纳入，需要回到该模型的论文或模型卡确认。

### 66.1.3 参数量是报告口径，不是唯一物理量

参数量先是一个计数：有多少模型权重值被纳入统计。它不说明这些值保存成什么格式、分布在哪些设备，也不说明前向时执行了多少算术。比如量化会改变每个权重的存储字节数，但不会因为一个 BF16 权重变成 INT4 表示，就把它变成四分之一的“参数个数”。

来源也可能分别报告 total、activated、non-embedding 等数字。这些字段的分母不同，必须保留字段名和来源，不能把它们揉成一个模糊的“模型大小”。

## 66.2 从一层 MoE 算到整模型

在 MoE 中，router 先为当前 token 计算各专家的分数，再选择本层要执行的专家。专家输出经路由权重组合后进入后续计算。由于每一层都有自己的 router，一个 token 在不同 MoE 层可以走过不同专家集合。

记第 $\ell$ 个 MoE 层第 $e$ 个专家的参数量为 $P_{\ell,e}$，该层共有 $E_\ell$ 个专家，token $x$ 在该层选出的专家集合为 $S_\ell(x)$。以下公式只表达“按一套固定的计数约定 $C$ 统计”，并不预设所有模型报告的 $C$ 相同。

### 66.2.1 Total：把全部 expert 权重计入

令 $N_{\mathrm{shared}}(C)$ 表示统计约定中单独归入共享部分的参数，$N_{\mathrm{other}}(C)$ 表示其他纳入计数的参数。三类参数按约定 $C$ 划成互不重叠的集合；遇到 tied weights 时按来源说明的去重规则处理。这样，整模型的总量可以按专家层逐层清点：

```math
N_{\mathrm{total}}(C)
=
N_{\mathrm{shared}}(C)
+
\sum_{\ell=1}^{L_{\mathrm{MoE}}}
\sum_{e=1}^{E_\ell} P_{\ell,e}
+
N_{\mathrm{other}}(C)
```

这里每个专家都计入，不论某个特定 token 会不会选到它。如果各层专家数或宽度不同，就对每层、每个专家实际的参数量求和，而不是把一个平均 expert size 当成精确模型账本。

### 66.2.2 Active：按每层路由集合累加

对一个 token $x$，只把每层实际选中的专家加入该 token 的专家路径，再加上来源口径计入的共同路径：

```math
N_{\mathrm{active}}(x;C)
=
N_{\mathrm{shared,path}}(x;C)
+
\sum_{\ell=1}^{L_{\mathrm{MoE}}}
\sum_{e\in S_\ell(x)} P_{\ell,e}
+
N_{\mathrm{other,path}}(x;C)
```

$N_{\mathrm{shared,path}}$ 和 $N_{\mathrm{other,path}}$ 特意带有 path 标记，因为“哪些共享参数算进 active”也要依来源定义。若专家大小不同，不能只数被选中的专家个数；必须把实际选中集合中的参数量相加。

### 66.2.3 一个等大小专家模型的总量与逐 token 计数

下面用一个教学设定核对全模型与 token 路径的差别。假设有 4 个 MoE 层，每层 8 个等大小专家，每个专家有 $0.25$B 个参数；共同路径中按本例口径计入 $2$B 个参数；每个 token 在每层选择 2 个专家。忽略模型中其他细分差异，路由路径可写成：

| MoE 层 | 该 token 选中的专家集合 | 本层计入 active 的专家参数 |
| --- | --- | ---: |
| 第 1 层 | $S_1(x)=\{2,7\}$ | $2\times0.25$B |
| 第 2 层 | $S_2(x)=\{1,4\}$ | $2\times0.25$B |
| 第 3 层 | $S_3(x)=\{3,5\}$ | $2\times0.25$B |
| 第 4 层 | $S_4(x)=\{2,8\}$ | $2\times0.25$B |

同一个 token 在每个 MoE 层都选两个专家，但不同层的集合可以不同。总参数把 4 层中的 8 个专家全部计入；active 则只加上该 token 每层选中的 2 个专家和共同路径：

```math
N_{\mathrm{total}}
=
2+4\times8\times0.25
=
10\ \mathrm{B}
```

```math
N_{\mathrm{active}}
=
2+4\times2\times0.25
=
4\ \mathrm{B}
```

所以本例中，expert bank 的选中部分是 $2/8=25\%$，而全模型 active/total 是 $4/10=40\%$。这些数是为了讲清口径而设的教学值，不对应任何真实模型的参数披露。

### 66.2.4 k/E 只适用于等大小的 expert bank

如果同一层的专家大小相同，而且每个 token 固定选 $k$ 个，那么该层专家库中被选中的参数比例是 $k/E$。这个比例只描述专家库本身。例如上一例的每层专家比例为 $2/8$，而全模型比例还受到共同路径影响。

多层模型要先算每层，再合并全模型计数。若专家大小不等、每层 top-k 不同，或某些层走 dense 路径，就不能用一个全局 $k/E$ 代表整模型。正确做法是保留每层参数量和路由规则，并说明来源报告的 Active Parameters 有没有包含共同路径。

## 66.3 每 token 活跃参数、请求内并集与存储字节

单 token active 不能回答一个请求会触及多少不同专家，也不能回答权重占多少字节。要回答这两个问题，需要分别记录 token 集合的并集和指定存储格式。

### 66.3.1 路由集合按层、按 token 变化

在同一层中，router 根据输入表示为专家打分；不同 token 可以得到不同的 $S_\ell(x)$。当一个 token 继续进入下一 MoE 层时，新的隐藏状态进入下一 router，得到新的专家集合。top-k 说的是“每个 token 在每个相关层选几个”，不是“整个模型只存在几个活跃专家”。

这也是为什么不能把模型简化成一个全局固定的专家子集。每 token 的 active 数可以稳定，而专家身份随 token 和层改变。

### 66.3.2 一批 token 的专家并集可以更大

对一个 token 批次 $B$，第 $\ell$ 层至少被某个 token 选中的专家集合是：

```math
U_\ell(B)=\bigcup_{x\in B}S_\ell(x)
```

只看专家库时，这个批次触及的不同专家权重为 $\sum_{e\in U_\ell(B)}P_{\ell,e}$。它可能远大于单个 token 的 $kP$，因为不同 token 可能走向不同专家；但这个并集也不等于每个 token 都执行了所有这些专家。请求或批次中的并集描述的是“哪些专家至少被用到”，逐 token active 描述的是“一个 token 的路由路径”，两者是不同账本。

### 66.3.3 参数个数乘精度字节只估原始权重载荷

若 $N$ 个参数都按同一种格式存储，每个参数占 $b$ 字节，可以先估原始权重载荷：

```math
M_{\mathrm{raw}}\approx N\times b
```

例如 10B 个参数若全按 BF16 的 2 bytes/parameter 存储，原始权重载荷约为 $20$ GB（按十进制单位）。这不是 GPU 显存估算：量化的 scale/zero-point、padding、混合精度、运行时 workspace、KV/state、activation、复制与 allocator 开销都尚未计入。若权重按设备分片，各卡持有的字节也不同于集群全量。

checkpoint 的 tensor 排列方式也不等于运行时的参数统计。Hugging Face Transformers 团队说明，其 MoE 加载路径可以把 checkpoint 中分别保存的 expert tensors 转换并打包到 runtime tensor；这是该 Transformers 加载实现的行为，不是参数个数发生变化的定义。[MoE 权重加载说明](https://huggingface.co/blog/moe-transformers)

## 66.4 用公开模型看数字，不混成同一口径

下面的数字保留各自来源的表述和结构信息。它们说明发布方怎样报告 total 与 active，不是把不同模型的参数统计重新统一审计后的精确值。

| 模型与来源 | 报告的 Total / Active | 来源同时披露的路由信息 | 阅读时保留的边界 |
| --- | --- | --- | --- |
| [Mixtral 8x7B 论文](https://arxiv.org/abs/2401.04088) | 47B total / 13B active | 每层 8 个 feed-forward experts；每 token 每层选 2 个 | 两个数是论文报告值；不能只用 $2/8$ 精确重算整模型 Active |
| [DeepSeek-V3 技术报告 v2](https://arxiv.org/abs/2412.19437v2) | 671B total / 37B activated per token | 除前三层外使用 MoE；每个 MoE 层有 1 个 shared expert 和 256 个 routed experts，每 token 选择 8 个 routed experts | 整模型计数还涉及共享路径和非专家部分；不应仅用 $8/256$ 推导报告值 |
| [Qwen3-30B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30.5B total / 3.3B activated | 同页还列 29.9B non-embedding、128 个 experts、8 个 activated experts | 页面提供了多个统计字段；记录访问时页面和模型 revision，不能仅由字段名称推断所有包含规则 |

### 66.4.1 Mixtral：top-2 不等于整模型的 active 比例

Mixtral 的论文说明，每层包含 8 个前馈专家，router 对每个 token 选择 2 个。这里的 top-2 关系解释了每个 token 为什么只经过一部分 expert bank；它并不意味着整模型 active/total 必须恰好是 $2/8$，因为全模型还包含共享和非专家部分，参数数也按论文报告口径汇总。[Mixtral of Experts](https://arxiv.org/abs/2401.04088)

### 66.4.2 DeepSeek-V3：共享 expert 与 routed experts 都要看

报告 §4.2 说明除前三层外采用 MoE；每个 MoE 层包含 1 个 shared expert 与 256 个 routed experts，每 token 选择 8 个 routed experts。这个结构能解释 active 路径由共享与选中专家共同组成，但不能只用 8 除以 256 推出表中的整模型报告值，因为它还绑定其他模型层和统计范围。[DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2)

### 66.4.3 Qwen3：同一模型卡并列 total、activated 与 non-embedding

Qwen3-30B-A3B 的官方模型卡把 total、activated、non-embedding 和 expert 数量并列披露。“30B-A3B”是便于识别的模型名，不应替代卡片里的字段。non-embedding 数量也提醒我们：参数统计可能对 embedding 另作口径；若拿模型卡、论文表格或模型名横向比较，先确认每列数字的定义。模型卡是访问时的在线页面，尚未固定 revision。[Qwen3-30B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)

### 66.4.4 发布数字留在各自版本与定义下比较

比较数字时，至少记录模型 ID、论文版本或模型权重 revision、访问日期、字段原名和来源给出的包含范围。再核对 shared expert、router、embedding、output head、tied weights、每层专家数及 top-k 是否披露。来源没有说明的部分标为 unknown，而不是用相邻字段、模型名或专家比例补齐。

## 66.5 参数数字不等于算力、内存、延迟和质量

Total 与 Active 是有用的架构统计，但不是完整资源模型，也不能说明模型质量。要从参数量推到其他结论，必须加入相应的计算口径、存储格式、系统实现或质量评估。

### 66.5.1 Active Parameters 和 FLOPs 有关联但不是同一量

在每个专家大小相同、每个 token 执行相同形状矩阵乘的条件下，选中 $k$ 个专家会让专家矩阵的计算量随 $kP$ 增长。Stanford 课程讲义按乘法和加法分开计数，给出一层选中专家部分约 $2kP$ FLOPs per token 的近似，并明确 router 和通信会另加成本。[课程讲义 §1.1](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf)

这是特定线性层与计数约定下的近似，不是整模型 FLOPs 公式。Attention、dense/shared layers、router、序列长度、矩阵形状和 FLOP 计数方式都会改变总计算量；训练还包括反向传播与优化器状态。第五册第 13 章负责训练计算量与预算估算，本章的 active 数字不能替代那套账本。

### 66.5.2 Total Parameters 和 GPU 显存不是同一量

Total 参数量乘以指定格式的每参数字节数，只能估算原始权重载荷。模型分片后各设备驻留的权重不同；量化元数据、scale、对齐与 packing 会影响权重字节；实际运行还需要 KV/state、activation、workspace 和缓存等内存。即使某个 token 的 active 数很小，也不能据此断言 GPU 只需要容纳这些 active 权重。

反过来，resident weight bytes 是特定 checkpoint、精度、分片和 runtime 下的部署状态，不能当作架构固有的第三种参数量。比较时把“参数个数”“模型文件字节”“整机/单卡峰值显存”分开记录。

### 66.5.3 延迟要看路由执行和 workload

每 token 选择较少专家，并不单独保证端到端更快。路由分发和结果聚合、跨设备通信、专家矩阵的实际批量、权重布局、prefill/decode 形状与 kernel 实现都参与执行时间。课程讲义将 router 与通信列为额外成本；Transformers 团队的文章也把 expert backend 和 expert parallelism 作为单独的实现问题讨论。[Stanford MS&E 319 课程讲义](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf) · [Transformers MoE 实现文章](https://huggingface.co/blog/moe-transformers) 这些材料支持“active 参数不足以预测端到端速度”，不构成跨设备或跨 runtime 的通用速度公式。更细的训练和服务代价交由第五册训练章节、第二十二章及推理系统书册展开。

### 66.5.4 总参数不能单独预测模型质量

Total 描述模型中可用的参数容量，Active 描述一个 token 的路径口径；它们都不能单独推出验证集 loss、任务能力或用户体验。2026 年一项 arXiv v1 MoE 配置研究报告，在超过 2,000 次预训练、约 10M–300M active parameters、总参数最高 6.6B 的实验中，对每步 active FFN 参数作匹配后，增加 inactive experts 参数在作者所测设置中改善了 held-out loss。[Slicing and Dicing: Configuring Optimal Mixtures of Experts v1](https://arxiv.org/abs/2605.11689v1)

这项结果提醒我们，active 参数相同的模型仍可能因 total expert capacity 不同而产生不同结果；它不是“加专家一定更好”的普遍规律，也不能直接外推到数百亿或数千亿规模、其他数据配比和其他训练预算。质量结论应引用具体任务和评估，不由一个参数数字代替。

## 66.6 审核并呈现模型参数数字

遇到参数量声明时，读者应能复述“它在统计什么”，而不只是记住一个数字。把来源和统计字段写全，缺少的条件就保持未知；存储、计算和质量分别报告。

### 66.6.1 先记录来源、模型 revision 与计入范围

核对具体模型 ID、论文版本或模型卡 revision、权重/配置 revision、访问日期及发布方原字段。进一步查阅是否计入全部专家、shared expert、router、embedding、output head 与 tied weights，以及 Active 是 per token 还是其他范围。只看到模型名或二手摘要时，不能把缺少的统计规则补成确定事实。

### 66.6.2 用分栏报告比较，不从缺失字段反推

一份可审阅的参数简报可以分开列出：

| 字段 | 应记录的内容 |
| --- | --- |
| Total Parameters | 发布方原数字、定义和来源版本 |
| Active Parameters | 原字段名称、是否按 token 统计、报告披露的共同路径/专家范围 |
| 结构字段 | 每层 expert 数、top-k、shared expert、dense/MoE 层分布 |
| 其他计数范围 | embedding、output head、router、tied weights、non-embedding 的说明或 unknown |
| 存储/运行 | precision/quantization、原始权重载荷估算、单卡/集群驻留显存或实测值，并注明如何测得 |
| 性能/质量 | FLOPs、tokens/s、延迟与质量指标各自的 workload、硬件、版本、benchmark 和日期 |

回答面试中的“一个 MoE 模型到底有多大”，先区分 total 与 active，再说明每 token 的路由和发布方的计数范围，最后指出显存、吞吐、延迟和质量还需独立口径。数字缺失时保留 unknown；不要把 Active 当作模型文件大小或一张 GPU 的容量承诺。
