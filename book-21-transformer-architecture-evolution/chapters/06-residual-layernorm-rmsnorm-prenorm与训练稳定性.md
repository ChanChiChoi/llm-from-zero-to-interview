# 第六章：Residual、LayerNorm、RMSNorm、Pre-Norm 与训练稳定性

## 6.1 为什么每个子层都要留一条残差路径

第 5 章讲过 attention 与 FFN 怎样处理 token 表示。现在看它们的输出如何进入下一层。Transformer 中，attention 和 FFN 通常都接在一条 residual stream（残差流）上：子层读取当前表示，产生一个与输入同宽的更新，再把更新加回主路径。

把一个子层先写成函数 $F$ ，输入为 $x$ ，残差连接的基本形式是 $x+F(x)$ 。在 Transformer block 中，attention 与 FFN 各自都有这样的连接；它们的输入输出宽度相同，才能逐元素相加。

这条路径解决一个直接问题：分支不必从零产生整个新表示，而可以把变化加到已有表示上。若分支暂时没有学到有用更新，没有后置归一化的结构会沿主路传递 $x$ ；Post-LN 则还会对相加结果做归一化。残差连接最初在 ResNet 中用于让深网络学习相对输入的残差映射；Transformer 将同一结构用于 attention 和 FFN 子层。[ResNet 原论文](https://arxiv.org/abs/1512.03385)和 [Transformer 原论文](https://arxiv.org/abs/1706.03762)分别给出了这两种应用。

## 6.2 残差连接把“完整映射”拆成“输入加更新”

如果目标变换与当前表示相近，网络只需让 $F(x)$ 学到两者的差异。例如输入为向量 $[2,-1]$ ，某子层给出的更新为 $[-0.2,0.3]$ ，相加后得到 $[1.8,-0.7]$ 。这个例子只展示加法关系；真实的 $F$ 通常是 attention 或 MLP，不是固定的增量表。

对不含归一化的残差映射 $y=x+F(x)$ ，局部导数可写成：

~~~math
\frac{\partial y}{\partial x}=I+J_F,
~~~

其中 $I$ 是恒等映射的 Jacobian， $J_F$ 是分支 $F$ 的 Jacobian。反向传播时，梯度既会经过 $F$ ，也会经过恒等路径。因此残差连接给深层网络提供一条较短的信息和梯度路径。

这不是稳定性保证。若多个分支的更新持续过大，主表示仍可能随深度放大；若训练过度抑制分支，模型也可能无法充分利用子层。残差连接要和归一化位置、初始化、学习率等一起理解。

## 6.3 LayerNorm：对单个 token 的 hidden 维做归一化

在标准 Transformer 中，LayerNorm（层归一化）对每个 token 的 hidden 维单独计算统计量。对一个维度为 $D$ 的向量 $x$ ，先求均值和方差：

~~~math
\mu=\frac{1}{D}\sum_{i=1}^{D}x_i,\qquad
\sigma^2=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2.
~~~

再标准化并应用可学习的逐维缩放与偏移：

~~~math
\mathrm{LN}(x)_i=
\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
~~~

用 $x=[1,3]$ 看统计过程。忽略很小的 $\epsilon$ ，均值是 $2$ 、方差是 $1$ ，标准化结果是 $[-1,1]$ 。若 $\gamma=[1,1]$ 且 $\beta=[0,0]$ ，输出仍是 $[-1,1]$ ；训练后的 $\gamma$ 和 $\beta$ 可以改变输出，所以不能保证归一化后的最终向量均值恰为 $0$ 、方差恰为 $1$ 。

归一化轴说明了它和 BatchNorm 的差异。标准 Transformer 的 LayerNorm 对每个 batch 中的每个 token 分别计算 hidden 维统计量，不把其他 token 或 batch 样本混入这次计算；BatchNorm 则依赖 batch 统计。[Layer Normalization 原论文](https://arxiv.org/abs/1607.06450)定义了按单个样本计算的归一化方式，[Stanford CS224N 讲义](https://cs224n.stanford.edu/readings/cs224n-self-attention-transformers-2023_draft.pdf)进一步明确了 Transformer 中 token 与 hidden 维的对应关系。

## 6.4 RMSNorm：控制向量尺度，但不减去均值

RMSNorm（均方根归一化）使用向量的 root mean square（RMS）控制尺度：

~~~math
\mathrm{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon},\qquad
\mathrm{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}.
~~~

标准形式不减去均值，通常也没有 LayerNorm 中的逐维偏移 $\beta_i$ 。对上一节的 $x=[1,3]$ ，令 $\gamma=[1,1]$ 并忽略 $\epsilon$ ，RMS 为 $\sqrt{5}$ ，输出约为 $[0.447,1.342]$ 。结果仍有非零均值，说明 RMSNorm 执行的是尺度归一化，不是均值中心化。

[RMSNorm 原论文](https://arxiv.org/abs/1910.07467)提出了“均值中心化可能不是 LayerNorm 成功所必需”的假设，并给出只按 RMS 缩放的形式。RMSNorm 的统计计算较简单，但具体运行速度还取决于算子融合、数据类型、张量形状和硬件，不能只凭公式断定某个实现一定更快。[LLaMA 论文](https://arxiv.org/abs/2302.13971)报告对子层输入做归一化并采用 RMSNorm；该模型案例见[第 54 章](./54-LLaMA现代开源LLM基线.md)。

## 6.5 Pre-LN 与 Post-LN：归一化在残差加法的哪一侧

LayerNorm 或 RMSNorm 是“做什么变换”；Pre-LN 和 Post-LN 是“在子层与残差加法周围怎样摆放归一化”。先用单个子层 $F$ 表示 attention 或 FFN。Post-LN 把归一化放在残差加法之后：

~~~math
h_{\ell+1}=\mathrm{Norm}\bigl(h_\ell+F_\ell(h_\ell)\bigr).
~~~

Pre-LN 把归一化放在子层输入处，残差加法后不再立即归一化：

~~~math
h_{\ell+1}=h_\ell+F_\ell\bigl(\mathrm{Norm}(h_\ell)\bigr).
~~~

| 结构 | 子层接收什么 | 残差加法后 | 读者应注意 |
| --- | --- | --- | --- |
| Post-LN | 当前表示 $h_\ell$ | 对加法结果做归一化 | 归一化也位于残差路径的输出处 |
| Pre-LN | 归一化后的 $h_\ell$ | 保留加法后的表示 | 子层输入受控，恒等分支仍直接连到下一层 |

原始 Transformer 在每个子层外先相加再做 LayerNorm，即 Post-LN。LLaMA 论文报告对子层输入做归一化，给出了 Pre-LN 风格结构的公开实例。Pre-LN 与 Post-LN 是不同 block 结构，不能把它们和 LayerNorm/RMSNorm 当作同一条架构轴。[Attention Is All You Need §3.1](https://arxiv.org/abs/1706.03762)、[Xiong 等人的结构图与定义](https://arxiv.org/abs/2002.04745)给出了对应形式。

## 6.6 为什么 norm 放置会改变梯度路径和 warmup 证据

把归一化函数记作 $N$ ，并暂时把它们看成可微函数。对 Pre-LN 映射 $h+F(N(h))$ ，局部 Jacobian 含有显式恒等项；对 Post-LN 映射 $N(h+F(h))$ ，归一化函数位于整个加法结果之后：

~~~math
J_{\mathrm{Pre}}=I+J_FJ_N,\qquad
J_{\mathrm{Post}}=J_N(I+J_F).
~~~

这里 $J_N$ 和 $J_F$ 分别是当前输入处的局部 Jacobian。这个比较说明了 Pre-LN 为什么保留较直接的残差梯度分量；它并不单独决定任意深度下的完整梯度大小，因为各层 Jacobian 会连乘，参数和输入也会变化。

Xiong 等人用 mean-field 分析研究初始化时的梯度，并在 IWSLT14、WMT14 翻译和 BERT 预训练设置中比较两种结构。他们报告：在其分析条件下，Post-LN 输出附近的部分参数期望梯度较大；Pre-LN 的初始化梯度较平稳，并且论文中的 Pre-LN 实验可在不使用 warmup 时达到可比结果。这解释了 Post-LN 的 warmup 依赖为何可能更强，但证据绑定论文的初始化、任务、optimizer 和训练设置，不能推出所有现代 LLM 都可移除 warmup。[论文原文](https://arxiv.org/abs/2002.04745)。

Warmup 是训练初期逐步提高学习率的策略。一种线性形式是：

~~~math
\eta_t=\eta_{\max}\frac{t}{T_{\mathrm{warmup}}},\qquad
1\le t\le T_{\mathrm{warmup}},
~~~

其中 $\eta_{\max}$ 是 warmup 结束时的学习率， $T_{\mathrm{warmup}}$ 是 warmup 步数。较小的早期学习率会让参数更新从小到大，而不是一开始就使用峰值学习率；warmup 本身不修复错误的初始化、数据或优化器配置。

## 6.7 为什么只看梯度大小不足以解释训练稳定性

训练不稳定不只有“梯度太大或太小”这一种解释。[Liu 等人的 Admin 研究](https://aclanthology.org/2020.emnlp-main.463/)认为，Post-LN 与 Pre-LN 的训练差异不能只归因于梯度不平衡：如果模型对 residual branch 依赖很强，小参数扰动就可能造成较大的模型输出变化；如果依赖过弱，模型又可能难以发挥子层作用。

Admin（Adaptive Model Initialization）通过初始化控制训练早期对 residual branch 的依赖，再让模型在训练后期逐渐释放能力。论文在英德、英法等机器翻译设置中报告了稳定性和性能结果，包括一个 72 层实验；这说明初始化和残差依赖值得一起检查，不是所有模型都应照搬同一初始化方案。

## 6.8 Pre-LN 之后还有哪些稳定性设计

Pre-LN 改善了若干设置下的初始化梯度路径，但研究者仍在处理层间梯度差异、极深网络更新幅度和 residual stream 的增长。下表中的方案针对不同问题，实验结论都应绑定来源条件。

| 方法 | 改动重点 | 论文报告与适用边界 |
| --- | --- | --- |
| [NormFormer](https://arxiv.org/abs/2110.09456) | 针对 Pre-LN 的层间梯度幅度失配，加入 attention 后 LayerNorm、head-wise scaling 和 FFN 中间 LayerNorm | 在 125M—2.7B 的 causal/masked LM 设置中报告预训练和下游任务改善；额外操作会影响训练时间与内存，结果不是对所有模型的保证 |
| [DeepNorm / DeepNet](https://arxiv.org/abs/2203.00555) | 缩放 Post-LN 残差连接，并配套缩放部分权重初始化 | 作者在多语机器翻译中展示了最多 1,000 层的模型；不能直接把该深度结论迁移到现代 decoder-only 预训练 |
| [Peri-LN](https://proceedings.mlr.press/v267/kim25u.html) | 在子层输入和输出处都放置归一化；Kim 等人将这种已有做法归纳并命名为 Peri-LN | 2025 年论文报告 400M—3.2B 模型实验，每个模型训练 300 亿 token、使用 5 个随机种子；后续 [Kan 等人的稳定性分析](https://arxiv.org/abs/2510.09904)以离散时间最优控制研究 norm placement 与 residual-step scaling，并在 GPT-2、OpenWebText、H200 设置中实验。两篇论文的假设和实验范围分别适用 |

Kim 等人 2025 年的论文将已有的子层外围归一化做法系统命名为 Peri-LN，并报告了其特定预训练设置中的方差、梯度和收敛结果。Kan 等人 2026 年更新的预印本是后续理论分析：在论文的最优控制建模条件下，Pre-LN 的隐藏状态可能无界增长，而 Peri-LN 呈受控增长；其 GPT-2 系列实验使用 OpenWebText，并在 H200 上运行。两项工作各自的实验和假设都不能推出所有模型都应改用某种 norm placement。总体上，稳定训练没有一个可脱离模型深度、初始化、optimizer、学习率、数据与精度的“最佳 norm”。

## 6.9 真实训练中怎样读稳定性信号

若有真实训练 trace，可以沿几个不同层次观察：

1. loss 是否突然上升、出现 NaN/Inf，问题从哪个 step 开始。
2. 各层 residual stream、子层输出和 activation 的尺度怎样随层数与训练步变化。
3. 梯度范数、参数更新范数与学习率曲线是否在异常 step 同时变化。
4. optimizer 状态、数据 outlier、混合精度溢出和分布式同步是否提供了其他解释。

令 $u_\ell$ 表示当前 block 配置实际产生的 residual-branch 输出；在 Pre-LN 中，它是 $F_\ell(\mathrm{Norm}(h_\ell))$ ，在 Post-LN 中，它是 $F_\ell(h_\ell)$ 。分支输出相对输入流的范数比可以写成 $r_\ell=\lVert u_\ell\rVert_2/(\lVert h_\ell\rVert_2+\epsilon)$ ，用于查看各层的分支输出尺度。它不等于参数更新与参数本身的比例，也没有适用于所有模型的固定通过阈值。单独一个比值或一次 loss spike 都不能判定根因；比较前还要固定模型版本、训练阶段、数据、精度、optimizer 和学习率 schedule。

## 6.10 常见误区

- **“Residual 会自动让任意深网络稳定。”** 它提供恒等路径，但多个分支连续写入仍可能累积过大；初始化和残差缩放会改变这一过程。
- **“LayerNorm 和 RMSNorm 都把输出变成均值零、方差一。”** LayerNorm 之后还有可学习 affine 参数；RMSNorm 根本不减去均值。
- **“Pre-LN 就不需要 warmup。”** Xiong 等人的结果针对指定初始化、任务与训练设置；现代模型的 warmup 决策还依赖 optimizer、batch、精度和数据。
- **“梯度不平衡就是唯一的不稳定原因。”** Admin 和后续方法还研究 residual branch 对参数扰动的敏感度、层间梯度差异、深度方向的状态增长与更新尺度。
- **“某论文的深度或质量结果能直接移植。”** 先核对架构、任务、初始化、训练数据和测量条件；不同论文回答的问题不完全相同。

## 6.11 自测与本章小结

1. 用自己的话说明 $x+F(x)$ 如何保留输入，并写出它的局部 Jacobian 中哪一项对应恒等路径。
2. 对 $x=[1,3]$ 手算 LayerNorm 和 RMSNorm 的结果，指出哪个步骤会减均值。
3. 写出 Post-LN 和 Pre-LN 方程，并说明为什么 Pre-LN 的局部 Jacobian 含显式恒等项。
4. 说明 Xiong 等人的 warmup 结果覆盖什么任务、初始化与证据边界；解释为什么不能据此判断所有现代训练配方都无需 warmup。
5. 比较 Admin、NormFormer、DeepNorm 与 Peri-LN 分别修改了什么，并为每种方案指出一个实验边界。

本章的主线是：Residual 保留输入主路径，LayerNorm 与 RMSNorm 以不同方式调整每个 token 的特征尺度；Pre-LN 与 Post-LN 改变归一化在残差路径中的位置，从而影响局部梯度路径和训练初期行为。Admin、NormFormer、DeepNorm 与 Peri-LN 说明稳定性还会受到初始化、残差分支依赖、层间差异和深度影响。读论文结论时，应把这些架构选择与任务和训练设置一起核对。

下一章转向位置编码：Self-Attention 怎样获得 token 顺序与相对位置的信息，见[第 7 章](./07-position-encoding总览-sinusoidal-learned-relative-rope-alibi.md)。
