# 第五章：从概率到模型行为：ML 与 LLM 的基础

大模型的许多现象，最后都可以追溯到几类基础问题：模型在表示什么概率，训练目标究竟奖励了什么，优化器如何改变参数，架构把什么信息放在一起，数据和 tokenizer 如何改变学习难度，以及评估是否真的测到了想测的能力。

这些问题在面试中常常以一句很短的话出现：“交叉熵是什么？”“为什么 attention 要除以平方根？”“大模型参数这么多，为什么不会完全过拟合？”如果只背定义，回答会在追问第一层就断掉；如果只背论文，又容易把一个特定实验的结论说成普遍规律。本章把它们重新放回一条连续的因果链：

~~~text
字符串
    -> tokenizer
    -> token 序列
    -> 条件概率
    -> Transformer 表示
    -> next-token loss
    -> 优化器更新
    -> 泛化、记忆和生成行为
    -> SFT 与偏好学习改变行为
~~~

初学者可以先把每一节理解成一个问题的完整答案：它解决什么问题，公式每个符号是什么意思，代码或例子如何体现它，什么时候会失效。专家则应继续问：这个结论依赖什么数据分布、tokenizer、优化预算和评估协议，观察到的变化能否归因于它，是否有更便宜或更可靠的替代方案。

第 4 章已经讨论如何把这些概念写成可靠的张量程序，本章重点是理解程序背后的数学和建模假设。第 6 章将进一步讨论数据管道、并行训练、故障恢复和成本，本章不会把那些系统实现压缩成清单。

## 5.1 先建立一张可解释的基础地图

### 5.1.1 六个主题其实是一个系统

概率回答“模型试图描述什么”；优化回答“参数怎样朝目标移动”；泛化回答“训练数据上的规律能否迁移到未见样本”；Transformer 回答“上下文怎样被表示和混合”；训练目标回答“预训练和后训练分别要求模型做什么”；tokenizer 回答“原始文本怎样被切成模型实际看到的单位”。

它们不能互相替代。例如，知道交叉熵的公式，不代表知道 labels 应该如何 shift；知道 Transformer 的结构，不代表知道右侧 padding 时为何不能无条件取最后一列 logits；知道 perplexity 下降，不代表知道两个不同 tokenizer 下的数值是否可比。

### 5.1.2 同一概念要经过三个层次

理解一个基础概念至少要经过三层。

第一层是直觉：它解决了什么问题。如果没有这一层，公式会变成符号记忆。

第二层是机制：它如何计算，输入和输出是什么，公式中的每个变量如何对应到模型。没有这一层，概念无法落到代码。

第三层是边界：它依赖什么假设，什么现象不能由它单独解释，如何设计实验区分可能原因。没有这一层，容易把相关性说成因果性。

例如，“交叉熵越低越好”只在同一个 tokenizer、同一套标签、同一个 reduction 和相近的数据分布下才有有限意义。它不能单独证明事实性、帮助性或安全性提升。

### 5.1.3 一张基础概念账本

| 概念 | 它描述的对象 | 不能直接推出的结论 |
| --- | --- | --- |
| token NLL | 真实目标 token 的平均负对数概率 | 任务成功、事实正确或用户满意 |
| perplexity | 在特定 tokenization 下的指数化平均 NLL | 跨 tokenizer 的绝对能力排序 |
| validation loss | 对保留分布的拟合程度 | 线上长尾和安全行为 |
| benchmark 分数 | 一个评估协议下的任务结果 | 所有场景的泛化 |
| attention 权重 | 某次前向中的加权系数 | 该 head 的完整语义解释 |
| 参数规模 | 模型可训练参数的数量 | 活跃计算、质量或成本一定更高 |
| tokenizer 压缩率 | 文本被表示成 token 的长度关系 | 对所有语言和任务公平 |

这张账本的用途不是限制表达，而是提醒读者把指标放回测量对象。面试中的基础问题，往往正是在考察这种边界意识。

## 5.2 概率链式法则：语言模型到底在建模什么

### 5.2.1 从一个序列开始

将文本经过 tokenizer 后得到长度为 T 的 token 序列。记 `x_{1:T}` 为从第 1 个到第 T 个 token，`x_{<t}` 为位置 t 之前的 token，`theta` 为模型参数。语言模型希望描述这条序列的联合概率：

~~~text
x_1, x_2, ..., x_T
~~~

自回归语言模型建模的是整个序列出现的概率。概率链式法则把联合概率分解为：

~~~math
P_\theta(x_{1:T})
=
\prod_{t=1}^{T}
P_\theta(x_t \mid x_{<t})
~~~

这个分解不是某个神奇的神经网络技巧，而是任意联合分布都成立的概率恒等式；语言模型选择从左到右，是为了让训练目标和生成过程使用同一个条件概率接口。

当模型给每个位置一个词表分布时，生成可以重复执行：

1. 读取当前上下文 x_{<t}；
2. 计算下一个 token 的分布；
3. 选择或采样 x_t；
4. 把 x_t 加入上下文，进入下一轮。

训练使用真实历史 token，称为 teacher forcing；推理使用模型自己刚刚生成的 token。两者的条件分布接口相同，但历史来源不同，这正是训练和生成之间产生差异的重要原因。

### 5.2.2 为什么只预测下一个 token 也能学习复杂能力

next-token prediction 的监督信号看起来很局部：每次只问“下一个 token 是什么”。然而一个正确的下一个 token 往往依赖很长的上下文。要预测代码中的闭合括号，需要保持结构；要预测代词，需要追踪指代；要预测一段证明或函数的后续，需要学习前文建立的约束。

因此，“模型只是在续写”并不等于“模型没有内部结构”。模型会学习哪些结构，取决于数据是否包含它们、模型容量是否足够、训练是否覆盖相关上下文，以及评估是否能区分机械记忆和新组合。

反过来，next-token 目标也没有直接要求模型验证事实、遵守安全规范或在不确定时拒答。只要某种续写在训练分布中具有较高概率，它就可能被模型偏好。这为后文讨论幻觉、SFT 和偏好学习埋下原因。

### 5.2.3 条件概率的三个边界

第一，概率是相对于 tokenizer 和上下文定义的。相同字符串换一个 tokenizer，序列长度和每个条件事件都会改变。

第二，模型的概率不是现实世界的真值概率。它是参数化模型对训练和推理分布的估计，受数据重复、来源偏差、模型校准和提示方式影响。

第三，高概率不代表高价值。一个流畅但错误的答案可能比一个谨慎的拒答更符合训练中的续写模式。需要事实检索、工具校验、外部评估或后训练目标来补足这些要求。

## 5.3 最大似然、NLL 与交叉熵

### 5.3.1 从最大化概率到最小化损失

给定包含 N 条训练序列的训练集 `D = {x^(1), ..., x^(N)}`，最大似然估计希望选择参数 theta，使观测到的训练序列概率最大。记 `M` 为所有有效目标 token 的位置集合；它排除了 padding、被 mask 的 prompt token 和其他不参与训练的位置。

~~~math
\theta^\star
=
\arg\max_\theta
\sum_{n=1}^{N}
\log P_\theta(x^{(n)})
~~~

使用对数是因为概率连乘会非常小，而 log 可以把乘法变成加法，便于数值计算和梯度优化。将目标取负，就得到负对数似然：

~~~math
\mathcal{L}_{\mathrm{NLL}}
=
-
\frac{1}{N}
\sum_{n=1}^{N}
\log P_\theta(x^{(n)})
~~~

展开序列概率后，语言模型的训练损失就是有效目标 token 的平均负对数概率：

~~~math
\mathcal{L}_{\mathrm{token}}
=
-
\frac{1}{|M|}
\sum_{(n,t)\in M}
\log P_\theta
\left(
x_t^{(n)} \mid x_{<t}^{(n)}
\right)
~~~

`M` 已在上文定义为有效目标位置集合，因此 padding、被忽略的 prompt token 或其他无效位置不进入分母。

这里还要区分两种常见的归一化。按序列平均时，每条序列先得到一个 NLL，再对 N 条序列平均；按 token 平均时，所有有效位置共享一个分母 `|M|`。当样本长度不同，这两种 reduction 会给长短样本不同的权重，因此报告 loss 时必须说明采用哪一种。

### 5.3.2 one-hot 交叉熵为什么等于 NLL

对于一个词表大小为 V 的分类分布，真实标签分布 p 是 one-hot，模型分布 q_theta 的交叉熵为：

~~~math
H(p,q_\theta)
=
-
\sum_{i=1}^{V}
p_i\log q_{\theta,i}
~~~

如果真实类别是 y，p_y=1、其余 p_i=0，于是：

~~~math
H(p,q_\theta)
=
-\log q_{\theta,y}
~~~

这正是该位置的 NLL。对所有有效 token 求平均，就得到 causal LM loss。F.cross_entropy 通常将 logits 先进行稳定的 log-softmax，再计算目标类别的负对数似然；它不是一个与概率训练完全不同的目标。

### 5.3.3 为什么不用 accuracy 直接训练

accuracy 只关心最大概率类别是否正确。模型给正确答案 0.51 和 0.99，accuracy 都记作 1；模型给错误答案 0.99，则它是一个高度自信的错误，但 accuracy 只记作 0。accuracy 也不能直接提供适合梯度下降的平滑信号。

交叉熵则会连续地奖励正确类别概率上升，并对自信地错施加更大的惩罚。它适合训练，但不代表它是完整的质量指标。生成任务还需要格式、事实性、帮助性、安全性、长度和成本等其他评估。

### 5.3.4 label smoothing 会改变解释

如果真实分布不再是严格 one-hot，而是把一小部分概率质量分给其他类别，交叉熵不再只等于 -log q_y。这可以减轻模型对单一标签的过度自信，但也会改变 loss 的绝对值和梯度。比较使用与未使用 label smoothing 的实验时，不能把 loss 下降直接解释为模型能力提升。

同样，padding mask、assistant-only loss、按 token 还是按序列平均，都会改变 loss 的测量对象。任何报告中的 loss 都应同时说明 tokenizer、有效 token 定义和 reduction。

### 5.3.5 一个数值例子：同一组预测如何产生 NLL、交叉熵和 PPL

把抽象符号落到数字上。设词表为 `{A, B, C, D}`，一条样本有三个有效目标位置。模型在三个位置给出的概率如下，粗体表示真实目标 token：

| 位置 | A | B | C | D | 真实目标 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | **0.50** | 0.20 | 0.20 | 0.10 | A |
| 2 | 0.10 | 0.25 | **0.25** | 0.40 | C |
| 3 | 0.25 | **0.125** | 0.50 | 0.125 | B |

每一行的交叉熵，在 one-hot 标签下就是该行真实目标概率的负对数。使用自然对数时，三个位置的 NLL 近似为：

~~~math
\ell_1=-\log(0.50)\approx0.6931,
\qquad
\ell_2=-\log(0.25)\approx1.3863,
\qquad
\ell_3=-\log(0.125)\approx2.0794
~~~

平均 token loss 是：

~~~math
\mathcal{L}_{\mathrm{token}}
=
\frac{\ell_1+\ell_2+\ell_3}{3}
\approx1.3863
~~~

于是 perplexity 为：

~~~math
\operatorname{PPL}
=
\exp(1.3863)
\approx4.000
~~~

这里的 4 不是说词表只有四个 token，也不是说模型真的每一步都在四个候选之间均匀选择。它只是这三个真实目标概率的平均负对数所对应的指数化结果。第三个位置的真实概率最低，因此它对平均 loss 的贡献最大；把正确 token 的概率从 0.125 提高到 0.25，会让该位置的 NLL 减少 `log 2`，但不会让其他位置的错误同时消失。

这个例子也能说明 mask 为什么属于指标定义的一部分。若另有一个 padding 位置，它的 label 不应加入 `M`。假设错误实现把它当作第四个有效位置，并且模型在该位置给 PAD 的概率只有 0.01，那么平均 loss 会变成：

~~~math
\mathcal{L}_{\mathrm{wrong}}
=
\frac{0.6931+1.3863+2.0794-\log(0.01)}{4}
\approx2.1910
~~~

这个数变大并不表示模型突然变差，而是评估者把本来不属于任务的目标混进了分母。反过来，如果两个实验使用不同的有效 token 集合，即使都报告“平均交叉熵”，它们也不是同一个测量对象。

KL 也可以用一个很小的例子检查方向。设真实或教师分布为 `p=(0.75,0.25)`，模型分布为 `q=(0.50,0.50)`。则：

~~~math
H(p)\approx0.5623,
\qquad
H(p,q)=-0.75\log0.50-0.25\log0.50\approx0.6931
~~~

因此：

~~~math
D_{\mathrm{KL}}(p\Vert q)=H(p,q)-H(p)\approx0.1308
~~~

如果交换方向，结果一般不会相同：

~~~math
D_{\mathrm{KL}}(q\Vert p)
=
0.50\log\frac{0.50}{0.75}
+0.50\log\frac{0.50}{0.25}
\approx0.1438
~~~

数值例子不替代定义，却能暴露三个常见混淆：NLL 看真实类别的概率，PPL 是平均 NLL 的指数化，KL 的数值和行为取决于方向。实际训练中还要继续核对 reduction、label smoothing、temperature 和 probability/log-probability 的接口约定。

## 5.4 熵、交叉熵与 KL：方向决定行为

### 5.4.1 三个量的定义

离散分布 p 和 q 的熵、交叉熵以及 KL 散度分别为：

~~~math
H(p)
=
-
\sum_x p(x)\log p(x)
~~~

~~~math
H(p,q)
=
-
\sum_x p(x)\log q(x)
~~~

~~~math
D_{\mathrm{KL}}(p\Vert q)
=
\sum_x p(x)
\log\frac{p(x)}{q(x)}
~~~

它们满足：

~~~math
H(p,q)
=
H(p)
+
D_{\mathrm{KL}}(p\Vert q)
~~~

当 p 固定时，H(p) 是常数，所以最小化交叉熵等价于最小化 D_KL(p || q)。但 KL 不是对称距离，也不满足一般距离的所有性质；写成哪个方向，不是排版选择。

### 5.4.2 两个方向的直觉

D_KL(p || q) 使用 p 的样本区域来惩罚 q 没有覆盖真实分布的地方。在监督学习里，p 常常是标签分布，因此它要求模型给真实类别足够概率。

D_KL(q || p) 则从 q 自己采样的区域出发，惩罚模型把概率放到 p 不支持的地方。它们在优化中可能表现出不同的模式，例如一个更倾向覆盖多个模式，另一个更倾向集中到某些模式。不能脱离优化方向讨论“KL 更好”。

### 5.4.3 KL 在大模型中的三个用途

在知识蒸馏中，教师分布可以提供比 one-hot 标签更丰富的软目标；学生模型通常要在温度和 KL 方向明确的前提下逼近教师。

在偏好优化或 RLHF 中，reference model 常用于约束 policy 不要偏离原始能力分布过远。KL 惩罚的权重越大，策略越保守；权重越小，策略可能更充分地追逐 reward，也更容易出现偏移和 reward hacking。

在分布比较中，KL 可以描述两个离散分布的差异，但对长尾、零概率和估计误差敏感。真实系统中常需平滑、截断或改用其他距离，并报告估计方法。

### 5.4.4 稳定实现和输入约定

PyTorch 的 KL 接口通常要求调用方明确输入是 probability 还是 log-probability，以及 target 是否已经取 log。把普通 probability 误传给 log-probability 参数，数值上可能不立即报错，却会得到错误的目标。基础理解必须包括 API 的输入约定，而不是只会写 KL 的数学式。

## 5.5 Perplexity：一个有条件的语言模型指标

### 5.5.1 从平均 NLL 得到 perplexity

如果平均 token NLL 使用自然对数，perplexity 定义为：

~~~math
\operatorname{PPL}
=
\exp\left(
\mathcal{L}_{\mathrm{token}}
\right)
~~~

当平均 NLL 为 log 2，PPL 为 2；它可以被直观理解为模型在每一步面对的“等效选择数”。这个直觉只在固定词表、固定 tokenizer 和同一评估协议下有效。

PPL 不是在说模型真的只会在 V 个 token 中均匀选择。真实分布通常高度不均匀，PPL 是交叉熵的指数化摘要，而不是一个直接可观察的候选数。

### 5.5.2 为什么跨 tokenizer 比较会失真

一个 tokenizer 可能把同一段中文切成较多 token，另一个 tokenizer 可能把常见词或代码片段合并成较少 token。每个 token 的预测事件不同，平均 NLL 的分母也不同。即使两个模型对字符级内容的预测能力相近，PPL 也可能因为分词方式不同而差异很大。

要比较不同 tokenizer，至少要报告：

1. tokenizer 和词表版本；
2. 评估文本完全相同；
3. 是否按 token、字符、字节或词重新归一化；
4. special token、截断和 padding 规则；
5. 评估集是否存在训练污染。

### 5.5.3 低 PPL 不等于好助手

PPL 测量的是对真实文本 token 的概率，不直接测：

- 回答是否遵循用户意图；
- 引用是否支持结论；
- 数学结果是否正确；
- 工具调用是否安全；
- 不确定时是否诚实拒答；
- 长上下文中是否找到关键证据。

后训练可能让模型更适合对话，却改变通用文本分布上的 PPL。正确做法不是把 PPL 和所有任务混成一个总分，而是明确它在评估矩阵中负责哪一格。

## 5.6 优化：参数为什么会朝某个方向移动

### 5.6.1 梯度下降的局部近似

设 loss 是参数 theta 的函数 `L(theta)`，令 `Delta` 表示一次参数扰动。在参数附近做一阶泰勒展开：

~~~math
\mathcal{L}(\theta+\Delta)
\approx
\mathcal{L}(\theta)
+
\nabla_\theta\mathcal{L}(\theta)^\mathsf{T}\Delta
~~~

令 `eta>0` 为学习率，并取 `Delta=-eta` 乘以当前梯度，那么一阶项为负：

~~~math
\theta_{k+1}
=
\theta_k
-
\eta
\nabla_\theta\mathcal{L}(\theta_k)
~~~

这解释了最基本的更新方向，但不保证每一步都让真实 loss 下降。学习率过大时，高阶项可能主导；batch 梯度只是总体数据梯度的估计；非凸网络还有平坦区、鞍点和多个等价或近似等价解。

### 5.6.2 学习率不是越小越安全

学习率过大可能造成震荡、梯度爆炸或 loss spike；过小则会让训练在预算内没有走到有效区域。warmup 让初期更新从较小幅度开始，常用于大 batch、参数随机初始化和混合精度训练，但它不是所有发散问题的通用药物。

后期 decay 的作用是减小更新步长，使模型在已有较好区域内细化。cosine、linear、constant-with-warmup 等 schedule 的选择要和总步数、token 预算、继续训练还是从头训练的场景一起看。改变 schedule 后，不能只比较某一个 step 的 loss。

### 5.6.3 梯度、更新量和参数量要分开

一个参数的梯度很大，不一定意味着参数更新很大；Adam 会按一阶和二阶统计调整步长。一个模型参数量很大，也不等于每步有同样的有效更新；混合精度、稀疏激活、冻结参数和梯度累积都会改变资源和更新行为。

训练监控中常见的量包括：

- gradient norm：当前梯度整体大小；
- parameter norm：参数整体大小；
- update norm：本步参数变化大小；
- update-to-weight ratio：更新相对于参数的比例。

这些量联合起来，比只看 loss 更容易识别学习率异常、某层梯度消失或某个模块没有参与训练。

## 5.7 SGD、Adam 与 AdamW

### 5.7.1 SGD 的基线

给定第 k 步的梯度 g_k，带学习率 eta 的 SGD 更新为：

~~~math
\theta_{k+1}
=
\theta_k-\eta g_k
~~~

mini-batch 让 g_k 成为一小批样本上的随机估计。batch 越小，估计噪声通常越大；这种噪声可能帮助探索，也可能增加训练波动。SGD 的简单性使它成为理解其他优化器的基线。

### 5.7.2 Adam 的自适应统计

Adam 为第 t 步的梯度 `g_t` 维护一阶矩 m 和二阶矩 v；`beta_1`、`beta_2` 是两个衰减系数：

~~~math
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
~~~

~~~math
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
~~~

由于 m 和 v 从零开始，需要偏差修正。`epsilon` 是防止除零的数值稳定项：

~~~math
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}
~~~

参数更新近似为：

~~~math
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\widehat m_t}
{\sqrt{\widehat v_t}+\epsilon}
~~~

Adam 的直觉是：梯度长期方向由 m 平滑，梯度尺度由 v 调整。它不等于自动找到最优学习率；beta、epsilon、weight decay、schedule 和有效 batch 仍然需要实验。

### 5.7.3 AdamW 为什么要解耦 weight decay

如果把 L2 正则直接加到梯度中，Adam 的自适应缩放会同时作用于正则项，使不同参数的实际衰减受到梯度统计影响。令 `lambda` 为 weight decay 系数，AdamW 将衰减作为单独的参数操作：

~~~math
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta
\frac{\widehat m_t}
{\sqrt{\widehat v_t}+\epsilon}
~~~

这个公式是教学化写法，具体实现还会处理参数组、学习率 schedule、混合精度和 fused optimizer。

通常不对 bias、LayerNorm/RMSNorm 的 scale 以及其他一维统计参数施加同样的 weight decay，因为它们并非承担特征投影的普通矩阵权重。但“哪些参数排除”是训练配方的一部分，应查看代码和参数组，而不是假设所有 AdamW 默认相同。

## 5.8 Batch size、梯度累积和噪声

### 5.8.1 三种 batch size

训练中至少要区分：

- micro-batch：一次设备前向和反向实际处理的样本数；
- data-parallel batch：一次数据并行同步前每个副本处理的样本数；
- global batch：一次 optimizer 更新实际聚合的样本数或 token 数。

在简单的等长、无丢弃场景下：

~~~math
B_{\mathrm{global}}
=
B_{\mathrm{micro}}
\times
N_{\mathrm{data}}
\times
K_{\mathrm{accum}}
~~~

变长语言模型更应该以有效 token 数而不是样本数定义 global batch，否则一条长序列和一条短序列的贡献不同。

### 5.8.2 梯度累积近似而非无条件等价

当模型没有 batch-dependent 层、随机性能够正确控制、loss reduction 与全局 token 权重一致时，梯度累积可以近似大 batch 的梯度。实际中仍有几类差异：

1. 每个 micro-batch 的 padding 比例不同；
2. dropout 每次前向使用不同随机掩码；
3. 混合精度的舍入发生在不同归约路径；
4. 梯度裁剪可能按 micro-step 或 accumulation cycle 执行；
5. optimizer 和 scheduler 只在 global step 更新。

因此“梯度累积等于大 batch”必须附带条件。尤其在 token-level loss 中，简单地平均每个 micro-batch loss，可能让短 batch 与长 batch 权重不正确。

### 5.8.3 batch 增大后的收益会饱和

更大的 batch 通常减少每个样本梯度估计的随机噪声，但吞吐、显存、通信和泛化之间存在折衷。可以用梯度噪声尺度作为一种分析工具，但它不是一个跨模型、跨任务的固定常数。工程上应结合 step time、tokens/sec、validation loss 和单位能力成本测量临界区。

扩大 batch 时同步改变学习率，也不能只依据线性缩放规则。优化器状态、warmup token 数、数据顺序、序列长度和训练总 token 都可能使线性规则失效。

## 5.9 泛化、记忆与过拟合

### 5.9.1 训练误差和泛化误差

训练集上的低 loss 说明模型提高了对训练分布的拟合；泛化关心的是从同一目标分布中抽取、但训练时没有见过的样本。若训练分布与真实使用分布不同，validation loss 也只能提供有限证据。

过拟合的表现不只是一条 validation loss 曲线：

- 固定 benchmark 提升，但新来源样本不提升；
- 训练样本或罕见字符串能够逐字复现；
- 训练格式被学得很牢，任务内容却没有改善；
- 某个安全集合上拒答率变好，但正常请求的误拒明显上升。

小规模 SFT 或 DPO 特别容易在风格、模板和偏好数据上过拟合。预训练规模大并不让后训练自动免疫过拟合。

### 5.9.2 大模型为何既能记忆又能泛化

记忆和泛化不是互斥标签。重复样本、罕见序列、代码和个人信息更容易留下可复现痕迹；抽象规律、常见语法和组合模式则可能被迁移到新样本。模型还可能在同一任务中对熟悉片段依赖记忆、对新组合使用泛化。

判断二者需要设计对照：

1. 去重前后比较训练和 held-out 结果；
2. 对同一知识做词面改写、实体替换和结构变换；
3. 测试罕见字符串的逐字复现与语义保持；
4. 检查 benchmark 是否和训练数据或近邻网页重叠；
5. 在不同来源、时间和语言切片上报告结果。

“模型理解了”或“模型只是记住了”都不是只凭一条生成样例就能成立的结论。

### 5.9.3 Bias-variance 在大模型中的用法

教学上，固定输入 x 时的平方误差可以拆成偏差、方差和不可约噪声。令 `\hat f_D(x)` 是在训练集 D 上得到的预测，真实标签为 `y=f(x)+epsilon`，并令 `D'` 表示独立重复抽取的训练集，则：

~~~math
\mathbb{E}_{D,y}
\left[
\left(y-\hat f_D(x)\right)^2
\right]
=
\underbrace{
\left(
\mathbb{E}_D[\hat f_D(x)]-f(x)
\right)^2
}_{\text{bias}^2}
+
\underbrace{
\mathbb{E}_D
\left[
\left(
\hat f_D(x)-\mathbb{E}_{D'}[\hat f_{D'}(x)]
\right)^2
\right]
}_{\text{variance}}
+
\underbrace{\sigma_\epsilon^2}_{\text{noise}}
~~~

这个分解帮助建立直觉，但深度网络的非线性、分布外任务和优化路径使它不能直接给出每个 LLM 的精确数值归因。`epsilon` 的方差记为 `sigma_epsilon^2`。

在实践中，可以把它转化为实验问题：增加模型容量是否改善训练和 held-out；增加数据是否减少不同切片之间的波动；改变 seed 是否显著改变结果；加入正则化、早停或数据多样性是否减少敏感性。每个结论都需要相应的对照，而不是套一句“模型大所以方差小”。

## 5.10 Transformer：把上下文变成可计算的表示

### 5.10.1 从 RNN 和 CNN 的限制出发

RNN 逐步更新状态，天然保留顺序，但训练时存在串行依赖，长距离信息需要经过许多步传播。CNN 通过局部卷积并行处理，短距离模式高效，但扩大感受野往往需要更多层或更大的卷积范围。

Self-attention 让每个 query 直接和一组 key 比较，再从 value 中取信息。任意两个位置之间可以用一次 attention 交互，训练时不同位置可以并行计算。代价是标准 attention 的 score 矩阵随序列长度平方增长。

这不是说 Transformer 在所有场景都绝对优于 RNN 或状态空间模型。它是在长距离交互、并行训练、硬件矩阵乘法和可扩展优化之间取得了一个强折衷。

### 5.10.2 Q、K、V 的三个角色

给定 hidden 表示 X，线性投影得到：

~~~math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
~~~

query 可以理解为当前位置提出的检索条件，key 是每个位置可被匹配的索引，value 是匹配后真正被聚合的内容。这个类比有助于理解，但不能把 attention 当成一个保证正确的数据库检索器：权重是模型内部计算的连续数值，可能关注错误位置，也可能把多个证据混合。

scaled dot-product attention 为：

~~~math
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\mathsf{T}}{\sqrt d}+A
\right)V
~~~

Q 的形状通常为 [B, H, T, d]，K/V 的长度可为 S，score 的形状是 [B, H, T, S]。A 可以是加性 bias 或由布尔 mask 转换而来的屏蔽项。softmax 沿 key 轴 S 归一化，因为每个 query 要在可访问的 key 集合上分配权重。

### 5.10.3 为什么除以 sqrt(d)

假设 q_i 和 k_i 的分量独立、均值为 0、方差为 1，则点积：

~~~math
q^\mathsf{T}k
=
\sum_{i=1}^{d}q_i k_i
~~~

其方差随 d 近似增长。d 越大，未经缩放的 score 绝对值通常越大，softmax 更容易集中到单个位置，梯度变得不平衡。除以 sqrt(d) 让 score 的尺度大致回到稳定范围。

这是基于初始化和独立性假设的尺度解释，不是对训练后所有 hidden 的严格统计定理。实际 score 还受归一化、RoPE、低精度、权重尺度和 mask 影响。

### 5.10.4 causal mask 与 padding mask

因果 mask 控制信息流的时间方向：位置 t 不能读取大于 t 的 key。padding mask 控制 batch 对齐产生的无效位置：有效 query 不能把 padding 当成真实上下文。它们可以合并为一个广播到 score 的 mask，但来源、语义和测试方式不同。

如果不加 causal mask，训练时位置 t 可能直接看到真实的 x_{t+1}，loss 变得异常低；推理时却没有真实未来 token，训练和推理条件不一致。若只缺 padding mask，模型会把填充符号当作上下文，短序列的表示和 loss 可能被污染。

## 5.11 Multi-Head、GQA 和 MQA：表达能力与推理成本

### 5.11.1 多头的作用不是给每个 head 编一个故事

Multi-Head Attention 将 D 维表示分成 H 个 head，每个 head 使用 d = D/H 的子空间计算关系，再把结果合并。不同 head 可以学习不同的交互模式，但“某个 head 必然负责语法、另一个必然负责指代”只是待验证的解释，不是架构定义。

更多 head 会改变 head_dim、投影布局、内存访问和并行粒度。head 数应与模型配置、硬件对齐和训练结果共同决定，而不是只因为“头越多越细”就增加。

### 5.11.2 GQA/MQA 为什么能节省 decode 资源

在 decode 阶段，历史 K/V 会被每个新 token 重复读取。标准 MHA 为每个 query head 保存一组 K/V；GQA 让一组 query head 共享 K/V，MQA 让所有 query head 共享 K/V。若每个元素占 b 字节，KV cache 的字节数近似为：

~~~math
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{kv}}d b
~~~

`H_kv` 减少会直接降低 cache 容量和带宽压力，但共享 K/V 也改变了模型的表示自由度。质量、吞吐和显存需要在同一个 checkpoint 或可比实验中一起观察。

这说明“参数量”和“推理成本”不是一个轴。GQA 不一定显著减少所有投影参数，却可能显著减少 decode 的 cache 读写；MoE 的 active parameters 也不等于 cache head 数。

## 5.12 位置编码：顺序信息如何进入注意力

### 5.12.1 没有位置时，attention 接近置换等变

在不使用位置相关参数或因果 mask 的纯 self-attention 中，交换输入 token 的顺序会同步交换输出位置，这种性质称为置换等变；它没有为顺序提供语义。decoder 的 causal mask 本身也注入了部分顺序约束，但不能替代完整的位置表示。因此“狗咬人”和“人咬狗”仍可能缺少足够的顺序区分。

绝对位置 embedding 直接为位置提供向量；相对位置方法把距离或方向加入 token 间的分数；RoPE 旋转 q/k；ALiBi 直接给不同距离添加线性 bias。它们都是位置归纳偏置，但外推、cache 和实现方式不同。

### 5.12.2 RoPE 的相对位置直觉

RoPE 在二维子空间中对 q 和 k 按位置旋转。对位置 i 的角度 theta_i，二维旋转矩阵为：

~~~math
R(\theta_i)
=
\begin{bmatrix}
\cos\theta_i&-\sin\theta_i\\
\sin\theta_i&\cos\theta_i
\end{bmatrix}
~~~

旋转后的 q_i 和 k_j 的内积包含 theta_i - theta_j，因此相对距离可以进入注意力分数。具体实现还要约定维度配对方式、频率 base、position_ids 的 dtype 和 cache 中历史位置的处理。

RoPE 能否外推到更长序列，不能只看一个配置字段。训练长度、频率缩放、checkpoint 训练约定、attention kernel、长文数据和评估任务共同决定有效能力。接口能够接受更长索引，只证明程序路径可用，不证明模型在长上下文中仍能检索和推理。

### 5.12.3 Pre-Norm 和 Post-Norm 的差异

残差连接把子层输出加回输入。Pre-Norm 常写成：

~~~math
x_{l+1}
=
x_l+
F_l(\operatorname{Norm}(x_l))
~~~

Post-Norm 则把归一化放在残差相加之后：

~~~math
x_{l+1}
=
\operatorname{Norm}
\left(
x_l+F_l(x_l)
\right)
~~~

Pre-Norm 的残差路径更直接，深层训练中常较稳定；Post-Norm 的表示尺度和梯度路径不同，在特定配方下也可能有优势。不能把“现代 LLM 常用 Pre-Norm”说成数学上唯一正确。

## 5.13 LayerNorm、RMSNorm 与 MLP

### 5.13.1 归一化解决的是尺度问题

对最后一维 hidden 向量 x，LayerNorm 的均值和方差为：

~~~math
\mu=\frac{1}{D}\sum_{i=1}^{D}x_i,
\qquad
\sigma^2=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2
~~~

归一化后再使用可学习的 gamma 和 beta：

~~~math
\operatorname{LN}(x)
=
\gamma\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta
~~~

RMSNorm 只计算均方根：

~~~math
\operatorname{RMSNorm}(x)
=
\gamma\odot
\frac{x}
\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon}
~~~

LayerNorm 移除均值并缩放方差，RMSNorm 保留均值方向但控制整体 RMS。两者的差别不仅在计算量，还在归纳偏置。epsilon、归一化轴、参数 dtype 和归一化所在位置都可能影响结果。

### 5.13.2 MLP 负责逐位置的非线性变换

Attention 在 token 之间搬运信息，MLP 对每个位置独立地做非线性特征变换。一个简单的两层 MLP 是：

~~~math
\operatorname{MLP}(x)
=
W_2\phi(W_1x+b_1)+b_2
~~~

它不直接混合不同 token，但可以把 attention 聚合来的表示投影到更高维的中间空间，再压回 hidden 维。MLP 往往占据 Transformer 的大量参数，因此不能把它当作 attention 旁边的“小附件”。

### 5.13.3 SwiGLU 的门控路径

SwiGLU 将两条投影相乘：

~~~math
\operatorname{SwiGLU}(x)
=
W_{\mathrm{down}}
\left[
\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot
(W_{\mathrm{up}}x)
\right]
~~~

一条支路产生门控，另一条支路产生被调制的值。门控结构增加了表达能力和参数/计算取舍；中间维度应由具体模型配置决定，不能一律写成 hidden size 的四倍。

## 5.14 三类 Transformer 架构

### 5.14.1 Encoder-only

Encoder-only 模型通常允许一个位置同时看左右上下文，适合表示学习、分类、序列标注和检索。BERT 的 masked language modeling 训练目标随机遮盖部分 token，让模型利用双向上下文恢复它们。

它并非不能生成任何文本，而是其预训练目标和输出接口不是天然的左到右生成。把 BERT 和 GPT 的差异简化成“一个理解、一个生成”也不够准确；更准确的说法是，训练目标、attention 可见性和使用方式不同。

### 5.14.2 Decoder-only

Decoder-only 模型使用 causal mask 和 next-token prediction。给定上下文，模型不断预测后续 token，因此对话、代码补全和开放式生成都可以共享一个接口。GPT 系列的早期工作展示了无监督预训练表示如何迁移到语言理解任务，后续 decoder-only 模型在规模、数据和训练配方上继续扩展。

它的代价是每个 query 在训练中受 causal 可见性限制，推理时需要处理逐 token decode、KV cache 和停止条件。

### 5.14.3 Encoder-decoder

Encoder-decoder 把输入序列先编码成表示，decoder 再通过 cross-attention 条件生成输出，适合翻译、摘要和许多输入到输出任务。T5 将多种 NLP 任务统一为 text-to-text 形式，展示了这种接口的灵活性。

三类架构没有脱离任务和预算的绝对排名。选择时要看输入输出关系、是否需要双向表示、生成长度、部署路径、checkpoint 生态和数据目标。

## 5.15 Tokenization：模型实际看到的不是字符

### 5.15.1 三种粒度的取舍

Word-level tokenization 词表直观，但词表会很大，未登录词和形态变化处理困难。Character-level 词表很小，几乎没有未知词，但序列变长，长距离建模和计算成本更高。Subword 方法试图在词表大小和序列长度之间取得折衷。

BPE 从较小的符号单元开始，反复合并训练语料中高频的相邻符号；WordPiece 使用相似的子词思想但目标和训练细节不同；SentencePiece 把原始文本视为字符序列处理，可以不依赖预先分词器，适合多语言和包含空格的文本。

这些名称不能互换成“都是分词”。词表训练规则、特殊 token、空格处理、Unicode 规范化和编码边界都会影响最终 token 序列。

### 5.15.2 tokenizer 的 token 密度

对一批文本，设 token 总数为 `N_token`，字符数或字节数为 `N_char_or_byte`，可定义平均 token 密度：

~~~math
\rho_{\mathrm{tok}}
=
\frac{\text{token 数}}
{\text{字符数或字节数}}
~~~

分母必须明确。`rho_tok` 越大，表示同样文本需要更多 token；但这个比率受语言、代码、数字、URL、规范化和样本长度影响，不能用一个短句代表整种语言。

更多 token 会带来至少三种成本：

1. 相同上下文窗口承载的字符内容减少；
2. 训练和推理 token 数增加；
3. attention 与 KV cache 的序列维度变长。

tokenizer 也会改变监督粒度。一个词被拆成多个 token 后，模型可以在子词层面学习形态和拼写，但错误会沿多个位置累积；代码中的缩进、标点和空格若切分不合理，会影响补全行为。

### 5.15.3 tokenizer 与多语言、公平和安全

不同语言的 token 密度差异会让相同字符长度对应不同训练成本和上下文占用。若只看 token 数，可能把某些语言的样本误判成更长、更贵或更难。多语言评估要同时报告语言切片、字符/字节长度和 token 长度。

安全上，Unicode 同形字符、零宽字符、特殊控制符和异常规范化可能让人类看到的字符串与 tokenizer 看到的 token 序列不同。安全过滤、日志审计和模型输入应尽量在规范化规则明确的前提下进行，不能只按界面上的字符判断。

### 5.15.4 词表大小不是越大越好

更大词表可能降低常见文本的 token 数，但会增加 embedding、lm head、softmax 和分布式通信的参数与内存。更小词表则可能导致序列变长。最佳点取决于语言混合、代码比例、硬件、上下文和训练预算。

词表版本还是模型 checkpoint 的一部分。更换 tokenizer 后，token id 的语义、embedding 行、special token 和训练统计都可能不再兼容；不能把 tokenizer 当作推理前随手替换的预处理脚本。

## 5.16 特殊 token、padding 和模板

### 5.16.1 特殊 token 是协议字段

BOS 表示序列开始，EOS 表示生成可以结束，PAD 用于 batch 对齐，UNK 表示词表无法直接表示的输入；对话模型还可能使用 system、user、assistant 或轮次分隔 token。具体语义由 tokenizer 和训练模板决定，名字相同不保证行为相同。

EOS 没有学好，生成可能无限延长；PAD 进入 loss，会鼓励模型预测填充；模板版本不一致，会让训练时的角色边界和推理时不同；special token 被用户输入伪造，还可能混淆数据管道和安全策略。

### 5.16.2 右侧 padding 的最后一列陷阱

假设两条序列右侧 padding 到相同长度：

~~~text
[A, B, C, PAD]
[D, E, F, G]
~~~

直接取 logits[:, -1, :] 时，第一条序列取到的是 PAD 位置的输出，而不是 C 之后应生成的位置。批量生成通常使用左侧 padding，或者根据每条序列的 attention mask 找到最后一个有效位置。这个细节和模型是否支持 padding、position_ids 如何生成、cache 如何追加都有关系。

### 5.16.3 assistant-only loss 不是唯一正确的目标

SFT 常把 user/system token 的 labels 设为 -100，只对 assistant response 计算损失，因为目标是学习回答而不是复述输入。但完整序列 loss 也可能用于继续预训练或特定对话配方。真正重要的是：目标 token 的集合必须与训练目的匹配，并在评估中保持同样的模板和 mask 约定。

## 5.17 预训练、继续预训练、SFT 与偏好学习

### 5.17.1 预训练学习基础分布

预训练用大规模文本、代码、数学或多模态序列优化 next-token 或相应的生成目标。它提供词法、语法、事实关联、代码结构和大量模式，但不会自动把模型变成一个遵守指令的助手。

训练数据的来源混合决定模型接触到哪些分布。网页、书籍、论文、代码、数学、对话和多语言数据各自提供不同信号，也各自带来重复、版权、隐私、质量、污染和安全问题。模型规模不能完全弥补缺失或错误的数据。

### 5.17.2 继续预训练改变领域分布

继续预训练用领域数据继续优化相同或相近的语言模型目标。它适合让模型熟悉医学、法律、代码库或企业术语，但也可能造成灾难性遗忘、领域偏置和过拟合。

判断继续预训练是否值得，不应只看领域验证 loss。要同时测通用能力、目标领域任务、格式遵循、安全和事实性，并记录领域数据混合比例、训练 token 和基线模型。

### 5.17.3 SFT 学的是示范条件下的行为

监督微调使用输入和高质量目标回答，让模型提高目标回答的条件概率。它能改变回答格式、任务遵循、角色边界和风格，但效果依赖示范数据的正确性、多样性、模板和 loss mask。

SFT 不是把所有事实“写进模型”的唯一机制，也不是事实更新的可靠替代。需要频繁更新的知识通常应由检索、工具或继续训练等机制承担，具体取舍取决于新鲜度、延迟、隐私和部署条件。

### 5.17.4 偏好学习塑造选择，而不是凭空创造能力

RLHF、RLAIF、DPO 等方法使用偏好或比较信号，让模型在多个可能回答中更倾向 chosen。它们可以提高帮助性、风格和安全一致性，但也可能放大标注者偏差、长度偏差、过度拒答、迎合和 reward hacking。

因此，偏好学习后的改善要用能力、偏好、安全、事实性和拒答质量的多维评估确认。一个总 reward 或单一 win rate 不能证明模型在所有维度都变好。

## 5.18 幻觉：为什么高概率续写可能是错误的

### 5.18.1 训练目标与事实验证不是同一件事

语言模型训练的直接目标是提高训练分布中真实后续 token 的概率。它没有在每一步都访问现实世界，也没有一个普遍可靠的真值数据库。面对知识缺口、矛盾来源或问题前提错误时，模型仍可能生成语法流畅的续写。

幻觉可能来自：

- 训练数据中的错误、过时信息和互相矛盾；
- 参数记忆与当前问题不匹配；
- 提示要求模型必须回答；
- 采样使低概率分支进入后续上下文；
- 长上下文中关键证据没有被有效利用；
- 工具、检索或引用链路返回了错误信息。

这不是一句“模型不知道所以胡说”就能完整解释的现象。要定位原因，需要记录上下文、检索证据、工具结果、生成参数和答案中可验证的 claim。

### 5.18.2 降低幻觉需要改变信息链

降低幻觉可以从多个层次入手：

1. 改善训练和后训练数据，增加不确定性表达和拒答示范；
2. 使用检索或工具提供外部证据；
3. 对答案进行 claim 级校验和引用；
4. 使用结构化输出限制协议错误；
5. 对高风险动作增加人工或规则核验；
6. 在评估中分别测无证据回答、证据支持、拒答和工具错误恢复。

temperature 调低可能减少随机性，却不能修复错误的参数知识或错误的检索结果。把“更确定”误认为“更真实”，正是解码指标和事实性指标混淆的例子。

## 5.19 Scaling：模型、数据与计算预算

### 5.19.1 三个尺度不能分开看

扩大模型参数、训练 token 或计算预算，可能带来能力提升，但它们不是三个互相独立的旋钮。固定参数增加 token，可能让模型受益；固定 token 增加参数，可能让模型受益；如果只扩大一项而严重失衡，新增预算的收益会下降。

经验 scaling law 常用幂律近似表示验证损失与模型规模、数据规模和计算量的关系。令 `N` 为参数量，`D` 为训练 token 数，`C` 为计算预算，则可写成教学上的幂律形式：

~~~math
L(N,D,C)
\approx
L_\infty
+
A N^{-\alpha}
+
B D^{-\beta}
+
E C^{-\gamma}
~~~

`L_infty` 以及系数和指数由实验拟合。这个公式是经验模型，不是所有任务和架构的物理定律。它不能直接给出某个新模型的 benchmark 分数，也不能代替对数据质量和后训练的评估。

### 5.19.2 训练 FLOPs 只能做量级估算

对 dense decoder-only Transformer，令 `N` 为参数量、`D` 为训练 token 数，常见教学近似是每个训练 token 的计算量与参数量成正比：

~~~math
C_{\mathrm{train}}
\approx
kND
~~~

其中 `k` 是与前向、反向、激活、架构和实现有关的系数。实际成本还包括通信、重计算、padding、评估、checkpoint、数据读取和低利用率。MoE 要进一步区分 total parameters 与每个 token 激活的参数。

更大的模型不一定更便宜；更高的 GPU 峰值 FLOPs 也不一定转化为更高的有效 tokens/sec。必须测量 MFU、通信比例、序列利用率和故障重跑成本。

### 5.19.3 Chinchilla 式计算最优的含义

在给定计算预算下，参数和 token 的配比存在一个近似最优区域。Chinchilla 的公开实验强调，许多当时的模型相对训练 token 不足，增加数据可以比单纯扩大参数更有效。这个结论针对特定模型族、数据和训练设置，不能机械外推到所有后训练或推理场景。

当模型服务于知识密集任务时，数据新鲜度、质量和去重可能比 token 总数更重要；当服务于复杂推理时，后训练和 test-time compute 也会改变最优预算。预算问题必须写清优化对象：预训练 loss、某个能力、单位成功任务成本，还是上线收入。

## 5.20 从基础概念到一条可验证的推理链

### 5.20.1 一个 token 的完整路径

考虑一句输入文本。它先经过 tokenizer，得到 token id；embedding 把 id 映射成向量；位置机制注入顺序；每层 attention 让 token 读取可见上下文，MLP 做逐位置的非线性变换；lm head 把最终 hidden 映射到词表 logits；softmax 给出下一个 token 的条件分布；训练时真实 token 进入交叉熵，推理时被选择或采样的 token 进入下一轮。

这条路径上的每个接口都有自己的失败方式：

| 阶段 | 关键问题 | 典型错误 |
| --- | --- | --- |
| tokenizer | 文本如何切分 | 特殊 token、Unicode、长度统计不一致 |
| embedding | id 如何查表 | 越界、dtype 或词表版本不匹配 |
| position | 顺序如何编码 | cache position 重启、长序列外推失效 |
| attention | 哪些信息可见 | mask 方向、padding、score 尺度 |
| MLP/norm | 表示如何变换 | 归一化轴、epsilon、权重配置 |
| lm head | hidden 如何变成 logits | tied weight、词表形状、dtype |
| loss/decoding | 如何训练或生成 | shift、prompt mask、概率与 log 概率混用 |

### 5.20.2 训练现象要回到最小机制

如果 validation loss 上升，可能是过拟合、分布变化、数据污染检查错误或学习率问题；如果生成幻觉，可能是知识缺口、证据链错误、采样、长上下文利用或 SFT 偏好造成；如果多语言成本异常，可能来自 tokenizer 压缩率而不是模型本身更“笨”。

好的基础分析会先提出可区分的假设，再选择最小实验。例如：

1. 固定模型和解码参数，只替换检索证据，判断幻觉是否来自外部上下文；
2. 固定数据和模型，比较不同 tokenizer 的 token 数与字符级评估；
3. 固定训练 token，改变参数量，观察 scaling 趋势；
4. 固定评估样本，检查训练数据近邻，判断 benchmark 分数是否被污染；
5. 固定 prompt，比较全量和 cache 前向 logits，定位位置或 mask 错误。

## 5.21 从定义到模型分析

### 5.21.1 先确定测量对象，再写公式

一个基础概念要进入模型分析，第一步不是背出名称，而是确定它作用于什么对象：token 分布、参数、hidden 轴、attention score 还是评估样本。对象确定后，公式才有明确的输入和输出；变量定义清楚后，读者才能把公式与代码中的 tensor、mask 或统计量对应起来。

例如，KL 的解释必须交代 p 是谁、q 是谁以及方向；batch size 必须区分 micro-batch、global batch 和有效 token；perplexity 必须同时说明 tokenizer 和 log base。少掉其中任何一个条件，数字就可能失去可比性。

### 5.21.2 反事实能暴露机制边界

一个能支撑推理的例子，不只展示正确结论，还应说明改变一个条件后结论如何变化。下面这些反事实分别对应不同的机制边界：

- 把 causal mask 去掉，训练会泄漏未来；
- 把 padding 从 loss 中移除，平均 loss 会改变；
- 把 temperature 增大，分布更平但错误知识不会消失；
- 把 tokenizer 换掉，PPL 和上下文占用不能直接比较；
- 把 validation 集反复用于调参，验证结果也会被过拟合；
- 把 reference 从 DPO 中移除，目标就不再是相对偏好约束。

反事实之所以有价值，是因为它把“知道定义”推进到“知道哪个条件在起作用”。

### 5.21.3 证据等级决定措辞强度

论文定义、官方 API 语义、某个公开模型卡、团队实测和个人推断拥有不同证据等级。书面分析应让措辞与证据匹配：数学对象可以说“定义为”，API 行为可以说“文档规定”，论文结果可以说“该实验报告”，单个项目结果可以说“在这个设置下观察到”，尚未完成的因果解释则应明确标为“待验证假设”。

Transformer 原论文的复杂度不能直接当成所有现代 attention kernel 的实测吞吐，某个 benchmark 上的 PPL 或准确率也不能直接当成通用能力。严谨表达不是回避结论，而是把结论的适用范围、比较契约和还需补充的证据说清楚。

## 5.22 学习练习：把碎片连成模型

### 练习一：从序列概率推到 loss

选一个长度为 5 的 token 序列，写出联合概率的链式分解，标出第一个输入位置和最后一个 logits 为什么不参与标准 next-token loss。再加入一个 padding 位置，说明有效目标集合如何变化。

### 练习二：画一层 decoder

在纸上标出 hidden、q/k/v、score、mask、softmax、value 聚合、MLP、残差和 norm 的形状。然后解释把 H_kv 从 H 改小后，哪些参数和哪一部分 cache 会变化。

### 练习三：做一次 tokenizer 对照

选择相同的中文、英文、代码、数字和 URL 样本，使用两个 tokenizer 统计字符数、字节数和 token 数。不要只比较平均值，还要按语言和内容类型分桶，记录特殊 token 和 Unicode 规范化差异。

### 练习四：设计一个泛化实验

构造训练近邻、词面改写、实体替换和真正新结构四类测试。比较模型能否复现字符串、保持答案规则和迁移到新组合，并写出哪些证据支持“记忆”、哪些证据支持“泛化”。

### 练习五：解释一个幻觉

记录一次模型错误回答的完整输入、检索证据、工具结果、生成参数和答案 claim。分别提出数据、检索、上下文、解码和后训练假设，再设计能区分它们的最小对照。

### 练习六：建立基础概念账本

为交叉熵、PPL、attention 权重、validation loss 和 benchmark 分数分别填写“测量对象、分母或归一化、可比较条件、不能推出的结论”。这张账本应当能直接指导一次实验报告，而不是只服务于面试记忆。

## 5.23 资料与证据边界

本章使用不同类型的公开资料支持不同层次的论述。

1. PyTorch Cross Entropy：[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
2. PyTorch KL Divergence：[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html)
3. PyTorch Embedding：[https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
4. Attention Is All You Need：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
5. Improving Language Understanding by Generative Pre-Training：[https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
6. BERT: Pre-training of Deep Bidirectional Transformers：[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
7. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer：[https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
8. Decoupled Weight Decay Regularization：[https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
9. RoFormer: Enhanced Transformer with Rotary Position Embedding：[https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
10. SentencePiece: A simple and language independent subword tokenizer：[https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
11. Neural Machine Translation of Rare Words with Subword Units：[https://aclanthology.org/P16-1162/](https://aclanthology.org/P16-1162/)
12. Training language models to follow instructions with human feedback：[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
13. Scaling Laws for Neural Language Models：[https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
14. Training Compute-Optimal Large Language Models：[https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
15. SentencePiece 官方实现：[https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

PyTorch 文档用于支持 API 输入约定；原始论文用于支持 Transformer、GPT、BERT、T5、AdamW、RoPE、子词 tokenizer、InstructGPT 和 scaling law 的定义或实验结论。文中的概率等式和优化公式是标准教学推导，代码与示例用于解释，不是对某个 checkpoint 或岗位要求的实测承诺。

论文中的 benchmark 和 scaling 结果都依赖具体数据、模型、训练预算和评估协议。尤其是“模型更大”“token 更多”“PPL 更低”“偏好分更高”这些表述，只有在比较契约一致时才有意义。关于当前模型的能力，应回到模型卡、官方评估和可复现实验，而不能从架构名词直接推断。

## 5.24 结语：基础不是答案模板，而是可追溯的解释

理解 ML 与 LLM 基础，最终要能够沿着一条链解释模型行为：

~~~text
tokenizer 决定观察单位
    -> 概率链式法则定义序列目标
    -> 交叉熵把目标变成可优化损失
    -> 优化器改变参数和表示
    -> Transformer 混合并变换上下文
    -> 数据与训练预算影响泛化和记忆
    -> SFT/偏好学习改变回答行为
    -> 评估协议决定我们看见什么
~~~

初学者应先能说清每个概念解决什么问题，再把公式的符号、代码的 shape 和一个具体例子对上。专家则要主动指出 tokenizer、数据分布、训练阶段、精度、评估切片和成本等条件，区分原始论文事实、教学公式、工程经验和待验证假设。

当你能解释一个 loss 为什么下降、一个 PPL 何时可比、一个 attention 为什么看到了某个位置、一个 tokenizer 为什么让某种语言更昂贵，以及一个后训练目标可能带来什么副作用，基础知识才真正进入了工程判断。下一章将把这条判断链放入大规模训练系统，继续讨论数据、并行、稳定性、checkpoint 和成本。
