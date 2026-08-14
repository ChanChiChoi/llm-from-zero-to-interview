# 第九章：训练稳定性：从异常现象到根因定位

训练开始后，loss 曲线通常会缓慢下降，但它不会沿着一条光滑的直线前进。某个 batch 可能让 loss 突然升高，某次浮点运算可能产生 Inf，某个 rank 可能在 collective 通信前就已经退出。真正棘手的地方在于：日志里看到的往往只是最后一个症状，而不是最早发生的错误。

初学者可以先把训练看成一条因果链：数据被整理成 token 和 label，模型根据它们计算 loss，反向传播产生梯度，优化器更新参数，分布式运行时再把多个 rank 的结果同步起来。链条前面出现的错误，会在后面被放大。例如，label mask 多保留了一段 padding，可能先表现为 loss 口径异常，随后表现为梯度异常，最后才表现为模型效果下降。

专家需要再多问一步：这个指标究竟测量了什么？它的统计分母是什么？异常是在 forward、backward、optimizer step 还是通信阶段第一次出现的？如果不能回答这些问题，直接降低学习率或跳过 batch 只是暂时改变现象，并不能证明根因已经消失。

本章从一条异常训练曲线出发，逐层分析数据、监督区域、模型计算、梯度、优化器、数值精度、分布式状态和 checkpoint 恢复。每一节都会同时给出适合初学者的直觉、专家需要记录的证据，以及这些证据的解释边界。

## 为什么训练 debug 很难

大模型训练的问题通常不是单点问题。

同一个现象可能来自不同层：

```text
loss spike
  可能是坏数据
  可能是学习率过大
  可能是混合精度溢出
  可能是分布式同步异常
  可能是 checkpoint 恢复错误
```

所以训练 debug 的关键不是背几个修复技巧，而是建立一条能复盘的因果链。最有用的原则可以写成一句话：先定位问题第一次出现在哪一层，再决定如何修复；修复之后还要用同样的观测重新运行，确认旧现象消失且没有引入新的偏差。

训练系统可以分成数据、tokenization 和 sample 构造、loss mask、模型 forward/backward、优化器和学习率、数值精度、分布式通信、checkpoint 和恢复八个相互连接的层。后文会沿着这条链解释为什么同一个 loss spike 需要不同的证据来判断。

面试中可以把这条链当作回答框架，但它首先是工程上理解训练系统的方式，而不是一张只用于背诵的清单。

### 资料与本章范围

本章的 API 行为以 PyTorch 的 autograd 异常检测、梯度范数裁剪、自动混合精度和分布式训练文档为准；梯度裁剪与混合精度的机制还会结合原始论文说明。阈值、warmup 长度和恢复策略则属于工程配置，论文或框架文档通常不能替一个具体模型给出普适答案。因此，文中把“文档明确规定的行为”和“需要在目标模型上验证的经验规则”分开叙述。

本章聚焦训练稳定性排查的工程闭环：loss spike、NaN / Inf、grad norm、数据 batch、label mask、学习率、optimizer state、mixed precision、分布式 rank 状态和 checkpoint 恢复。生产监控平台、NCCL 深层故障诊断和完整的模型架构稳定性理论，只在它们影响上述判断时提及。

### 训练日志应该记录什么

每个 optimizer step 可以记录一组相互关联的观测量：


```math
h_t=(L_train,t,L_val,t,G_t,eta_t,S_t,T_t,R_t,U_t,Delta_rank,t)
```

其中 `L_train,t` 是训练 loss，`L_val,t` 是验证 loss，`G_t` 是全局梯度范数，`eta_t` 是当前学习率，`S_t` 是动态 loss scale，`T_t` 是本次更新真正参与监督的有效 token 数，`R_t` 是 tokens/sec，`U_t` 是 GPU 利用率或类似系统指标，`Delta_rank,t` 是 rank 间需要进一步解释的差异。把 `T_t` 放进日志很重要：两个 batch 的平均 loss 可能相同，但一个 batch 只有几百个有效 token，另一个 batch 有几万个有效 token，它们对参数更新的统计可靠性并不相同。

训练 loss 也应该明确它的分母。令 `m_j` 表示第 `j` 个位置是否参与监督，单个 batch 的 causal language modeling loss 可以写成：

```math
L_t=-\frac{1}{T_t}\sum_{j=1}^{n_t}m_j\log p_\theta(y_j\mid x_{<j}),
T_t=\sum_{j=1}^{n_t}m_j
```

这里的 `n_t` 表示 batch 展平后的 token 槽位数，`T_t` 才是有效 label token 数。若 `T_t=0`，这个平均值没有定义；工程上应记录并跳过该 batch，而不是把空监督 batch 当作 loss 为 0。

```math
t^\*=\min\{t\mid a_t=1\}
```

其中 `a_t` 是诊断程序根据日志生成的异常指示量。它不是一个神奇的总分，而是把有限性、有效 token 数、loss 相对变化、梯度范数和 rank 状态等信号统一到时间轴上：

```math
a_t=\max\left\{
\mathbf{1}[\neg \mathrm{isfinite}(L_{\mathrm{train},t})],
\mathbf{1}[\neg \mathrm{isfinite}(G_t)],
\mathbf{1}[T_t=0],
\mathbf{1}[r_t\ge \tau_{\mathrm{spike}}],
\mathbf{1}[\Delta_{\mathrm{rank},t}>\tau_{\mathrm{rank}}]
\right\}
```

公式中的 `r_t` 是后文用窗口中位数定义的 loss 比值，`Delta_rank,t` 是分布式部分定义的 rank 间差异，`tau_spike` 和 `tau_rank` 都是需要在基线数据上设定的阈值。`a_t=1` 只表示“值得保存现场并继续分析”，不表示已经找到了根因。比如 rank 间 loss 不同，可能是各 rank 拿到了不同难度的数据；只有在有效 token 数、数据切分和聚合方式一致时，差异才更像同步问题。阈值也必须和模型、数据混合比例、batch 大小及数值格式一起解释。

## 1. 先建立健康训练的基线

Debug 的前提不是先定义一个固定的“正常 loss”，而是先建立与当前模型、数据混合、有效 token 口径和硬件配置相匹配的基线。不同任务的 loss 绝对值可能差异很大；没有上下文的 2.0 既不能自动说明训练良好，也不能自动说明训练异常。

在初学者视角里，可以先观察下面几类趋势：

1. train loss 总体下降，允许小幅波动。
2. validation loss 同步下降或稳定改善。
3. learning rate 按 schedule 正常 warmup 和 decay。
4. grad norm 在合理范围内波动。
5. 没有持续 NaN / Inf。
6. tokens/sec 相对稳定。
7. GPU utilization 没有长时间掉到很低。
8. checkpoint 能保存并恢复。

如果你没有这些基线指标，出现问题时就很难判断异常来自哪里。

专家还会为每个指标保存计算口径。train loss 和 validation loss 是否按有效 token 加权，grad norm 是裁剪前还是裁剪后，learning rate 是按 micro-batch 记录还是按 optimizer step 记录，tokens/sec 是本 rank 数值还是全局数值，这些差异都可能让两条看似不同的曲线其实不可比较。基线因此不只是几条折线，也包括数据版本、随机种子、并行配置、精度配置和日志聚合方式。

## 2. 从异常曲线建立证据链

看到异常时，最有价值的问题不是“应该把哪个超参数改小”，而是“第一次异常发生在什么时间、什么计算阶段、什么数据上”。时间点越晚，前面的错误越可能已经被后续操作掩盖；先保存现场，才有机会把现象还原成因果链。

| 阶段 | 要回答的问题 | 需要保留的证据 |
| --- | --- | --- |
| 时间定位 | 第一个异常 step 是哪一个？发生在 micro-step 还是 optimizer step？ | 原始 loss、有效 token 数、micro-step、optimizer step、事件时间 |
| 层级定位 | 异常先出现在数据、forward、backward、更新还是通信？ | batch 元信息、logits/梯度有限性、参数更新、per-rank 日志 |
| 反事实缩小 | 换成健康 batch、单卡或较小模型后现象是否仍在？ | 固定 seed、最小复现配置、对照运行结果 |
| 修复验证 | 原异常消失了吗？吞吐、验证集和其他指标是否退化？ | 修复前后同口径曲线、checkpoint 和变更记录 |

这张表的意义在于让每次修改都对应一个假设。例如，把学习率减半是在检验“更新步长过大”，而不是在证明数据没有问题；把坏 batch 换掉是在检验“样本触发了异常”，也不是在证明新的数据过滤规则适用于全部语料。

因此不要一上来同时改学习率、batch size、数据、精度和并行策略。这样即使训练好了，也不知道是哪一个因素起作用。对于会消耗大量 GPU 时间的任务，可以先用固定的一小段样本、较小模型和单卡运行，确认假设后再回到原始规模验证；缩小规模改变了问题时，也要把它记录为诊断结果，而不是把小规模结果当作最终结论。

### 2.1 异常组合比单个指标更有信息量

单个指标很少能唯一确定根因。比如 grad norm 变大可能是学习率、数据或 loss 归一化造成的；但把 loss、grad norm、有效 token 数和数值有限性放在一起，候选解释会明显减少。下面的组合只是诊断起点，不能替代对具体 batch 的检查。

| loss | grad norm | 有效 token 与 rank 状态 | 更值得先验证的解释 |
| --- | --- | --- | --- |
| 有限但突然升高 | 正常 | batch 难度或来源改变 | 数据分布、样本模板、验证实现 |
| 升高且同步变大 | 同步变大 | 有效 token 正常 | 学习率、warmup、更新步长或异常 label |
| 变成 NaN/Inf | 变成 Inf/NaN | 单个 rank 先出现 | forward/backward 溢出、坏输入或某个 rank 的数据 |
| 训练正常、验证恶化 | 正常 | 验证集固定 | 过拟合、评估模式、数据污染或训练目标不匹配 |
| loss 正常但吞吐下降 | 正常 | rank 等待或利用率下降 | dataloader、通信、显存换入换出或 kernel 回退 |
| 只有一个 rank 异常 | 该 rank 异常 | 其他 rank 仍在 collective | rank 私有 batch、OOM、非法值或控制流分叉 |

初学者可以把这张表理解成“先看哪一层”；专家则会把每一行转换成最小反事实实验。例如，固定参数只替换 batch，可以检验数据假设；固定 batch 只切换精度，可以检验数值假设；固定单卡结果再扩展到多卡，可以检验通信假设。实验的价值在于排除候选原因，而不在于让某个指标暂时回到漂亮的范围。

## 3. Loss spike

Loss spike 是大模型训练中非常常见的异常。

表现：

```text
loss 突然升高
grad norm 同步变大
有时会恢复，有时进入 NaN
```

先把 spike 和 NaN 分开。一个有限的短时 spike 可能只是当前 batch 的难度或分布发生变化；loss 变成 NaN 则说明数值运算已经失去有限性。两者可以同时出现，但不能用“最后都不稳定”代替对它们的区分。还要区分训练集 spike 和验证集 spike：只有验证集升高时，优先怀疑数据分布、评估实现或过拟合，而不是立刻修改 optimizer。

### 3.1 可能原因

| 原因 | 说明 |
| --- | --- |
| 坏数据 batch | 极长、乱码、异常 token、错误 label |
| 学习率过大 | 更新步子太大，参数跳到不稳定区域 |
| warmup 太短 | 训练初期还不稳定就给了高 lr |
| 梯度爆炸 | grad norm 突然变大 |
| FP16 overflow | 数值溢出导致 loss 异常 |
| checkpoint 恢复不完整 | optimizer/scheduler 状态错位 |
| loss 分母或有效 token 数改变 | 同一批 token 的平均方式发生变化，曲线出现假 spike |
| 只在 validation 出现 | 评估数据、模型模式或分布发生变化，不一定是训练更新导致 |

### 3.2 用三个对照判断 spike 的来源

第一组对照是数据对照。保存 spike step 的 batch id、样本来源、长度、token 分布、特殊 token 和 label mask，并把同一批样本在健康 checkpoint 上重新跑一遍。如果旧参数在这批数据上也产生高 loss，问题更可能在样本、模板或标签；如果只有新参数产生高 loss，数据只是触发条件，根因要继续向优化器和数值状态寻找。

第二组对照是状态对照。把 spike 前后的 grad norm、learning rate、warmup 进度、loss scale、optimizer step 和 checkpoint 恢复时间放在同一条时间轴上。loss 和 grad norm 同时升高，通常比只有 loss 升高更支持“更新或监督信号异常”的假设；loss 升高但梯度、学习率和数值状态都平稳，则要优先检查 batch 难度和 loss 聚合口径。

第三组对照是验证集对照。训练 loss spike 而验证 loss 没有同步变化，可能是一个难 batch；训练和验证都在恢复 checkpoint 后跳变，则要检查状态是否错位；验证集单独恶化，则应检查评估模式、数据预处理和分布变化。排查的目标是区分这些模式，而不是把每一种模式都归结为学习率问题。

一个实用的 spike 判定不要只看单点 loss，而要和前面一小段窗口比较。设最近 `k` 个健康 step 的 loss 中位数为基线：

```math
r_t=
\frac{L_t}
{\mathrm{median}(L_{t-k},\ldots,L_{t-1})+\epsilon}
```

当下面条件成立时，可以先把该 step 标成候选异常点：

```math
r_t\ge \tau_{\mathrm{spike}}
```

其中 `tau_spike` 是 spike 阈值。把 2 到 3 倍作为初筛只能算一种可调的告警起点，不能当作跨模型、跨数据集的标准；窗口长度、loss 方差、有效 token 数和 batch 混合比例都会改变这个比值的解释。这个公式的作用是给日志定位一个一致口径，真正的因果判断仍要结合 batch、grad norm、lr 和 loss scale。

### 3.3 修复动作必须对应证据

如果 batch decode 后确认是乱码、错误模板或 label 错位，应修数据并重新生成样本；仅仅跳过这一个 batch，不能修复同源样本继续进入训练的问题。如果 loss spike 与 peak learning rate 或恢复后的 scheduler step 同步出现，应先检查学习率和 optimizer state，再决定是否降低 peak lr 或延长 warmup。

gradient clipping 适合限制一次过大的更新，但它不能修复错误的 loss 分母，也不能把坏数据变成好数据。FP16 出现 overflow 时，切换 BF16 可能缓解动态范围问题，但仍要检查 softmax、归一化、自定义 kernel 和 loss scaling 的行为。若参数已经被 NaN 污染，继续训练通常没有意义，应保存现场并从 spike 之前的可用 checkpoint 重新开始。

把这些证据组织成面试回答时，可以先说现象，再说最早 step 和对照实验，最后说明修复动作与验证方式：loss spike 要先判断是数据驱动、优化器驱动还是数值驱动，不能直接假设是学习率问题。

## 4. NaN 和 Inf

NaN / Inf 说明某个数值已经失去有限性，但它们出现的位置仍然决定排查方向。Inf 常见于溢出或除以极小数，NaN 则可能来自 Inf-Inf、0/0、无效的对数或已经被污染的输入。看到最终的 loss=nan 只说明错误已经传播到了 loss，不能据此断定错误起源于 loss 计算。

常见现象：

```text
loss = nan
grad_norm = inf
某些 parameter norm 变成 nan
optimizer state 出现 inf
```

### 4.1 常见原因

1. FP16 overflow。
2. 学习率过大。
3. warmup 太短。
4. 梯度爆炸。
5. attention logits 过大。
6. 输入中有异常值。
7. 自定义 loss 中除以 0。
8. 分布式 reduce 前某个 rank 已经 NaN。

### 4.2 找到第一个非有限张量

排查时要把一次更新拆成四个阶段：forward 输出和 loss 是否有限，backward 后梯度是否有限，unscale 和 clipping 后梯度是否有限，optimizer step 后参数与 optimizer state 是否有限。若 forward 已经产生 NaN，优先看输入、mask、logits 和自定义算子；若 forward 有限而 backward 变坏，优先看反向 kernel、梯度尺度和混合精度；若只有 optimizer step 后变坏，再检查学习率、权重衰减、optimizer state 和恢复状态。

同时保存第一个非有限值所在的 tensor 名称、rank、layer、dtype、shape 和 batch id。只记录“step 600 的 loss 是 NaN”会丢掉最重要的时间顺序；同一个 step 内，logits、loss、某层梯度和参数的先后关系才决定下一步实验。

PyTorch 的 `torch.autograd.detect_anomaly(check_nan=True)` 可以在调试阶段追踪产生异常梯度的反向节点，但它会显著降低速度，不能常驻大规模训练。梯度裁剪前还应先调用 `unscale_`，否则检查到的是被 loss scale 放大的梯度，而不是实际要交给 optimizer 的梯度。

有限性检查可以写成：

```math
F_t=(isfinite(L_t),isfinite(G_t))
```

只要 `F_t` 中有一个分量为 false，就不要继续让 optimizer 更新参数。应先保存现场、定位第一个坏张量，并视参数是否已被污染回滚到上一个可用 checkpoint。PyTorch 的 `clip_grad_norm_` 可以通过 `error_if_nonfinite=True` 把非有限的 total norm 直接暴露出来；这是一种调试保护，不是对根因的修复。

### 4.3 处理非有限值

处理动作要和发生阶段对应。输入或 batch 本身含有非法值时，应隔离样本并修正数据管线；学习率、warmup 或梯度范数异常时，应回到更新方程检查步长；FP16 溢出时，应检查 loss scaling、算子 dtype 和高风险 kernel，必要时比较 BF16 或局部 FP32 的结果；参数已经非有限时，应从参数仍然有限的 checkpoint 恢复。

“跳过异常 batch”也有边界：如果异常发生在 gradient accumulation 的中间，已经累积的部分梯度不能和下一个 batch 拼在一起继续更新，通常应丢弃整个 accumulation window，并记录被丢弃的有效 token 数。否则更新中混入了不完整的梯度，问题可能变得更难复现。

在面试中，NaN 的核心回答不是“换 BF16”，而是说明如何定位第一个坏掉的 step、阶段和张量，再根据证据选择数据、精度、超参或 checkpoint 处理。

## 5. 梯度爆炸和梯度消失

### 5.1 梯度爆炸

表现：

```text
grad norm 突然变大
loss spike
参数范数异常
可能进入 NaN
```

梯度范数是更新信号的整体尺度，但它不是“模型是否健康”的单一温度计。梯度大可能来自异常样本、错误的 loss 归一化或过大的更新步长；梯度小也可能只是有效监督 token 很少。要把它和参数更新幅度、分层梯度统计、有效 token 数及 optimizer step 一起看。

这些原因需要从更新方程和数据证据两边交叉验证。学习率过大或 warmup 不足会让很多层的更新同时放大；异常 batch 或 mask 错误通常会和特定样本、有效 label 数或某些层的梯度集中相关；初始化不稳定则往往从训练早期就能观察到，而不是只在某个随机 batch 突然出现。

处理时可以先用 clipping 防止一次更新继续污染参数，同时修复导致大梯度的根因。降低 lr 或延长 warmup 是对优化步长假设的实验；检查 batch、label mask 和 loss 的有效 token 分母，是对数据与目标假设的实验。每次实验都应保留同一个 checkpoint、同一段样本和同一组日志，否则不同结果无法归因。

全局梯度范数通常按所有可训练参数的梯度拼接后计算：

```math
G_t=
\sqrt{\sum_i \lVert g_{t,i}\rVert_2^2}
```

如果最大允许范数为 `c`，global norm clipping 可以抽象成：

```math
g'_t=
g_t
\min\left(1,\frac{c}{G_t+\epsilon}\right)
```

这说明 clipping 只在 `G_t>c` 时缩小梯度方向的长度，不会在正常梯度范围内改变更新。它改变的是梯度向量的尺度，而不是数据、目标函数或模型结构。

判断更新是否真的过大，还可以记录相对参数更新量：

```math
u_t=\frac{\lVert\theta_{t+1}-\theta_t\rVert_2}{\lVert\theta_t\rVert_2+\epsilon}
```

`G_t` 很大但 `u_t` 没有异常，可能是参数尺度本来就大；`G_t` 看起来正常但 `u_t` 突然升高，则要检查学习率、optimizer state 或参数分组。大模型还应按 layer 或 parameter group 记录梯度范数，否则一个异常小的 embedding/head 梯度可能被全局范数平均掉。

### 5.2 梯度消失或过小

表现：

```text
loss 几乎不下降
grad norm 很小
参数更新很小
```

可能原因：

1. learning rate 太小。
2. loss mask 把大部分 token 忽略了。
3. 模型某些参数没有参与计算。
4. optimizer 参数分组漏了参数。
5. mixed precision underflow。

“梯度很小”还要和“梯度为零”区分。前者可能是学习率小、模型处于平台期或有效监督弱；后者可能是参数没有参与计算、mask 把路径切断、detach 误用或某个分支没有被路由。混合精度 underflow 则可能让本来非零的数在低精度表示中变成零，应该通过局部 FP32、梯度统计和小规模对照来验证。

在面试中，梯度异常可以按四个量回答：全局与分层 grad norm、相对参数更新量、有效 label token 数、optimizer 参数分组。只有这四类证据互相支持时，才适合把问题归因于“梯度消失”或“学习率不合适”。

### 5.3 梯度累积时，哪个梯度才代表一次更新

当显存不足以容纳完整 batch 时，训练通常把多个 micro-batch 的梯度累积起来，再执行一次 optimizer step。此时“某个 micro-batch 的 grad norm”与“真正送入 optimizer 的 grad norm”不是同一个量。若第 `b` 个 micro-batch 有 `T_b` 个有效 token，按有效 token 归一化的累计梯度应近似写成：

```math
g_{\mathrm{update}}=\frac{\sum_{b=1}^{B}T_b g_b}{\sum_{b=1}^{B}T_b}
```

如果代码先对每个 micro-batch 求平均、再对 batch 平均，且各 `T_b` 差异很大，就会让短回答 batch 与长回答 batch 具有相同权重，改变实际训练目标。debug 时要同时记录 micro-step、accumulation boundary、每个 `T_b`、累计梯度范数和是否发生 clipping；否则一次 spike 可能被错误归因给某个单独 micro-batch。

梯度裁剪也有两个口径：每个 micro-step 裁剪，和累积完成后裁剪。前者会改变每个局部梯度的方向组合，后者只限制最终更新的整体尺度。两者都可能合理，但必须在实验记录中明确，因为它们不是等价实现。

## 6. 数据异常导致训练异常

数据问题是训练异常的高频根因。

常见数据异常：

1. 空样本。
2. 极长样本。
3. 乱码。
4. 错误编码。
5. 特殊 token 缺失。
6. label 全部为 ignore。
7. prompt 和 answer 错位。
8. 重复样本比例过高。
9. benchmark 或测试集泄漏。

对 SFT 或指令微调，异常 batch 的一个关键指标是有效 label 比例：

```math
T_{attn}=\sum_j a_j
T_{label}=\sum_j m_j
q_{label}=\frac{T_{label}}{T_{attn}+\epsilon}
```

这里 `a_j` 表示 attention mask，`m_j` 表示 label mask，`T_attn` 排除了 padding，`T_label` 只统计真正参与监督的 token。如果 `T_label=0`，这个 batch 没有定义良好的监督目标，应在 collator 或训练循环中显式处理；如果 `q_label` 很低，则要检查截断、角色模板和 assistant span 是否丢失。若 prompt token 也参与了 loss，模型可能学会复读 prompt。稳定训练不只要求 batch shape 正确，还要求有效监督区域正确。

### 6.1 把 batch 还原成可读样本

不要只打印 tensor shape。异常 batch 至少要能从日志追溯到来源，并能把 token 重新 decode 成人可以阅读的文本。建议保存异常 step 的：

```text
sample id
source
input length
input_ids 前后若干 token
decoded text
labels 中有效 token 数
attention_mask sum
loss mask span
```

对 packed sample，还要保存每个文档的边界和 segment id；否则 decode 后看起来是一段连续文本，却无法判断 attention 是否错误地跨过了文档边界。对多轮对话，要把角色 token、assistant span、截断位置和 special token 一起打印。很多所谓的“训练不稳定”其实在第一次 forward 之前就已经写错了监督目标。

数据检查还要按 source、长度桶、语言或任务类型分组统计。一个总体平均的 `q_label` 可能正常，但某个 source 的 label 全为空；总体 loss 也可能稳定，但极长样本或某一类合成数学题持续贡献 spike。按分组看分布，才能区分偶发 outlier 和系统性数据 bug。

很多问题一 decode 就能看出来，例如回答被截断、角色 token 错、全是 padding、文本乱码。

### 6.2 数据 bug 的典型现象

| 现象 | 可能数据问题 |
| --- | --- |
| loss 不下降 | label 错位、有效 label 太少 |
| loss 很低但模型很差 | 数据重复、泄漏、任务太简单 |
| loss spike | 异常 batch、极端长度或坏样本 |
| 模型学会复读 prompt | SFT label mask 错误 |
| 模型输出格式混乱 | chat template 不一致 |

## 7. Loss mask 和 label 错误

Loss mask 错误非常隐蔽。

训练脚本可能正常运行，loss 也会下降，但模型行为不对。

常见错误：

1. user token 参与 SFT loss。
2. assistant token 没有参与 loss。
3. padding token 参与 loss。
4. 多轮对话只训练最后一轮。
5. label 没有 shift。
6. 截断后只剩 prompt，没有 answer。

loss mask 的正确性可以拆成三个相互独立的问题：监督 token 是否足够、causal LM 的 shift 是否正确、padding 是否被排除。把它们合并成一个 0/1 分数会隐藏究竟是哪一项出错，因此更适合记录成诊断向量：

```math
D_mask=(q_label,Delta_shift,Delta_pad)
```

其中 `Delta_shift` 表示 causal LM 的 input / label 对齐关系，`Delta_pad` 表示 padding 是否被排除，`q_label` 表示有效监督比例。`q_label` 低不一定是错误，例如一个 prompt 很长、回答很短的任务可能天然如此；但 `Delta_shift` 或 `Delta_pad` 错误会改变训练目标本身，必须通过 decode 和小样本手算验证。

检查方式：

```text
decode input_ids
decode labels != -100 的部分
确认模型到底在学什么
```

这也是面试中很有辨识度的回答：训练 debug 时要 decode 一批 input 和 label mask，直接说明模型到底在学习哪些 token，因为很多格式问题只看 tensor shape 看不出来。

## 8. 学习率和 optimizer 问题

学习率不是一个孤立的数字，它和 optimizer 的状态、有效 batch、梯度归一化以及 scheduler 的计步方式共同决定参数更新。以 AdamW 为例，更新可以抽象为：

```math
\theta_{t+1}=\theta_t-\eta_t\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}-\eta_t\lambda\theta_t
```

因此看到 loss spike 时，既要看 `eta_t`，也要看偏置修正后的动量、二阶矩和权重衰减项。只看配置文件里的 peak lr，无法知道恢复后 optimizer state 是否仍然对应当前 step。

### 8.1 学习率过大

学习率过大通常表现为：

loss 和 grad norm 一起升高，更新量 `u_t` 变大，或者训练刚越过 warmup 的峰值就出现异常。训练初期不稳定更支持 warmup 或初始化问题，但不能只凭时间位置下结论；如果异常紧跟 checkpoint 恢复，则还要先排除 scheduler 重置或 optimizer state 错位。

降低 peak lr、延长 warmup 或启用 clipping，分别是在改变更新步长、改变步长增长速度和限制极端梯度。它们应当作为相互独立的实验，不应同时修改后再把结果归因给其中某一个。


### 8.2 学习率过小

学习率过小通常表现为：

loss 下降慢、相对更新量长期很小、验证指标不动，都可能来自学习率过小，但也可能是 mask 没有提供足够监督或参数根本没有进入 optimizer。应先比较 `T_t`、梯度和 `u_t`：梯度本身也很小，和只有更新量很小，是两种不同问题。

若确认是学习率或 scheduler 配置，应检查 scheduler 的总步数、warmup 步数、是否按 optimizer step 递增，以及梯度累积后实际 update 次数是否和配置一致。不要用 micro-batch 数代替 optimizer step 数，否则 warmup 和 decay 都会提前或延后。


### 8.3 optimizer state 恢复错误

如果从 checkpoint 恢复时只加载模型权重，没有加载 optimizer 和 scheduler，训练可能突然不稳定。模型权重描述“当前参数在哪里”，optimizer state 描述“优化器认为自己已经积累了什么”，scheduler state 描述“当前应该走到哪一个学习率阶段”；这三者必须对应同一个训练进度。

特别是 AdamW 的动量状态和学习率 schedule，如果不连续，loss 曲线可能跳变。恢复后应先打印 global step、consumed tokens、当前学习率、每个 parameter group 的学习率和 optimizer state 的 step，再用一小段固定 batch 做对照运行。若只加载权重是有意为之，那它应被当成一次新的微调实验，而不是“无缝继续训练”。

## 9. 混合精度溢出

FP16 的指数范围比 BF16 小，因而更容易在大激活或大梯度处 overflow，也更容易在很小的梯度处 underflow；BF16 的动态范围接近 FP32，但有效精度位数更少。两者不是简单的“谁更稳定”，还取决于硬件、kernel、归一化实现和模型中的数值范围。

表现：

1. loss scale 频繁下降。
2. grad norm 变 Inf。
3. loss 变 NaN。

处理时要先判断是范围问题、精度问题还是 kernel 问题。可以比较 BF16、FP16 加动态 loss scaling，以及高风险操作局部使用 FP32 的结果；同时检查 softmax、norm、自定义 kernel 和通信前后的 dtype。单纯把 dtype 改掉而不记录跳步次数，可能只是把异常推迟。


loss scaling 的核心关系是：

```math
\tilde{L}_t=
S_tL_t
```

```math
\tilde{g}_t=
S_tg_t
```

```math
g_t=
\frac{\tilde{g}_t}{S_t}
```

loss scaling 主要是把很小的梯度暂时放大，降低 FP16 下溢的概率；它不能修复错误的 loss，也不能保证放大后的值不会溢出。如果缩放后的梯度出现 Inf / NaN，scaler 通常会跳过本次 optimizer step 并调整 scale。调试时要记录 `S_t` 是否频繁下降、实际跳过了多少次 update，以及跳步是否和 loss spike、坏 batch 或 rank 异常发生在同一段时间。

使用 AMP 时，梯度裁剪应发生在 unscale 之后。典型顺序是：

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
scaler.step(optimizer)
scaler.update()
```

如果在 `unscale_` 之前裁剪，阈值会被 loss scale 一起放大，日志中的“裁剪前 grad norm”也会失去直观意义。BF16 通常不需要 FP16 式的动态 loss scaling，但仍需要有限性检查。

因此混合精度问题不只是 dtype 选择，也和 loss scaling、kernel 实现、梯度尺度以及 optimizer 是否跳过更新有关。

### 9.1 从数值范围理解 FP16 与 BF16

初学者常把“16 位浮点”理解成同一种数值格式，实际上 FP16 和 BF16 把位数分配给指数与尾数的方式不同：

| 格式 | 指数位 | 尾数位 | 最大有限值（约） | 最小正规数（约） | 主要风险 |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 | 5 | 10 | 6.55e4 | 6.10e-5 | 大激活 overflow、小梯度 underflow |
| BF16 | 8 | 7 | 3.39e38 | 1.18e-38 | 动态范围大，但舍入精度较低 |
| FP32 | 8 | 23 | 3.40e38 | 1.18e-38 | 显存和带宽开销更高 |

这张表只能说明表示范围，不能直接预测一个模型会不会稳定。实际计算中，autocast 可能让矩阵乘使用低精度、让 softmax 或归一化使用 FP32；优化器还可能保留 FP32 状态。专家排查时要记录参数 dtype、激活 dtype、累加 dtype、optimizer state dtype 和 kernel 实际路径，而不是只看配置文件中的 `torch_dtype`。

loss scaling 的边界也可以用同一张表理解：它把梯度暂时推离 FP16 的下溢区，但也可能把梯度推入上溢区，所以 scaler 需要根据非有限梯度回退 scale。若 scale 长期下降，说明当前数值范围或计算路径仍不适合；若 scale 长期不变，也不能证明没有数据或优化器问题。

## 10. 分布式同步错误

分布式训练中的 bug 通常更难定位，因为一个 rank 的本地错误可能表现为其他 rank 在 collective 处 hang 住。这里至少要区分两类问题：控制流不一致，例如某个 rank 提前异常、collective 调用顺序不同；数值或数据不一致，例如各 rank 的 batch、有效 token 数或 loss 归一化不同。前者首先表现为等待，后者首先表现为曲线差异。

常见现象：

1. 训练 hang 住。
2. 某些 rank loss 不一致。
3. 某些 rank OOM。
4. all-reduce 报错。
5. checkpoint 恢复后 rank 状态不一致。

### 10.1 先判断是控制流还是数据/数值差异

常见根因包括某个 rank 的数据长度不同、dataloader 没有正确 shard、gradient accumulation 边界不一致、collective 调用顺序不同，以及某个 rank 先 NaN 或 OOM。随机种子和 dropout 状态差异会让本地数值不同，但在正确的数据并行语义下，随机性不同本身不等于通信错误；需要先明确项目是否要求逐位确定性。

### 10.2 逐级缩小并记录每个 rank

先用单卡确认问题是否独立于通信，再用单机多卡区分进程间同步，最后才进入多机网络和拓扑。每个 rank 至少记录 global step、micro-step、batch id、有效 token 数、loss、grad norm、当前 collective 名称和最近一次成功的 collective。遇到 hang 时，主进程日志往往停在“等待”，真正有用的是找出没有到达同一 collective 的 rank。

每个 rank 的 loss 至少要做一致性观察：

```math
T_t=\sum_r T_{t,r}
L_t=\frac{\sum_r T_{t,r}L_{t,r}}{T_t}
\Delta_{\mathrm{rank}}=\max_r L_{t,r}-\min_r L_{t,r}
```

rank 间的 loss 差异需要结合有效 token 数解释。不同 rank 处理不同样本时，本地平均 loss 不必逐位相同；更有意义的是先用有效 token 数做全局加权，再观察差异是否超出基线。可以把每个 rank 的观测写成一个诊断记录，而不是合并成一个总开关：

```text
rank_observation = (local_loss, effective_tokens, finite, collective_position)
```

如果某个 rank 先变 NaN，其他 rank 可能只是被 collective 卡住。排查时要看 per-rank loss、batch id、有效 token 数和 collective 调用位置，而不是只看主进程日志；如果所有 rank 都在同一 forward 阶段出现非有限值，则通信更可能只是在传播模型或数据问题。

在面试中，分布式 debug 可以沿着单卡、单机多卡、多机逐级缩小，并明确区分数据/模型异常和 collective 控制流异常。

## 11. Checkpoint 恢复问题

Checkpoint 恢复错误很容易导致训练曲线异常。

完整恢复通常需要：

1. model weights。
2. optimizer state。
3. scheduler state。
4. scaler state。
5. random state。
6. dataloader position。
7. distributed state。
8. tokenizer 和 config。

常见错误：

1. 只恢复模型，不恢复 optimizer。
2. scheduler step 从 0 重新开始。
3. tokenizer 版本不一致。
4. FSDP/ZeRO checkpoint 加载方式不匹配。
5. 数据继续位置错乱。

完整恢复所需的状态可以写成一个集合；它描述恢复实验要携带哪些信息，而不是把所有恢复质量压缩成一个开关：

```math
C_resume={model,opt,sched,scaler,rng,data,distributed,tokenizer,config,step}
```

如果只加载了 model weights，就只能说“加载了部分状态”，不能说“按原实验连续恢复”。对训练恢复来说，optimizer 动量、scheduler step、loss scaler、随机状态、dataloader 位置和分布式切分都可能影响接下来的 loss 曲线。tokenizer、config 和参数命名也必须与权重语义一致，否则即使张量成功加载，训练目标也可能已经改变。

更严格的做法是在恢复后用固定的一小段数据做连续性对照：从 checkpoint 继续运行一步，与未中断运行保存的参考轨迹比较 loss、梯度和参数更新。若随机性被严格控制，可以比较参数差异；若只要求统计连续，则应比较窗口内的均值、方差和有效 token 口径，而不是要求每个浮点数逐位相同。

在面试中，checkpoint 能用于推理不代表能无缝恢复训练；应明确列出 optimizer、scheduler、scaler、随机状态、数据位置、分布式状态以及 tokenizer/config，并说明如何用连续性对照验证恢复结果。

### 11.1 checkpoint 还要证明自己没有写坏

即使字段齐全，checkpoint 也可能在写入过程中被中断。大文件分片上传、对象存储最终一致性、进程退出和磁盘空间不足，都可能留下“目录存在但某个 shard 不完整”的状态。训练系统通常应先写入临时目录，完成所有 shard、manifest、大小和 checksum 校验后，再通过原子重命名或版本指针把它发布为可恢复版本，同时保留前一个版本作为回退。

分布式 checkpoint 还要保存 shard 的拥有 rank、参数布局和 world size 相关元数据。加载时不能只按文件名猜测张量归属；FSDP、ZeRO 或自定义 sharding 的保存格式不同，错误的加载方式可能在形状上看似成功，却让参数和 optimizer state 的分片语义错位。

因此“能打开 checkpoint”只是文件可读性证据，“从中断点继续后曲线连续”才是恢复行为的证据。两者应分别记录：前者检查 manifest、大小、checksum 和版本；后者检查 step、token cursor、学习率、随机状态、有效 token 口径以及恢复后的一小段参考轨迹。

## 12. 最小可运行训练诊断 demo

下面的 demo 用标准库模拟一次训练诊断。输入是 step 级训练日志、batch 元信息、per-rank loss 和 checkpoint 字段；输出是 spike、非有限值、梯度异常、坏 batch、rank 差异、checkpoint 缺项和建议动作。它不模拟真实 autograd 或通信，只演示如何把不同来源的证据对齐到同一个 step。

```python
import math
from statistics import median

history = [
    {"step": 1, "loss": 2.40, "grad_norm": 0.8, "lr": 1e-5, "loss_scale": 4096, "tokens_sec": 980},
    {"step": 2, "loss": 2.18, "grad_norm": 0.9, "lr": 2e-5, "loss_scale": 4096, "tokens_sec": 1005},
    {"step": 3, "loss": 2.05, "grad_norm": 1.0, "lr": 3e-5, "loss_scale": 4096, "tokens_sec": 996},
    {"step": 4, "loss": 5.80, "grad_norm": 7.5, "lr": 4e-5, "loss_scale": 4096, "tokens_sec": 991},
    {"step": 5, "loss": 2.02, "grad_norm": 1.1, "lr": 5e-5, "loss_scale": 4096, "tokens_sec": 997},
    {"step": 6, "loss": float("nan"), "grad_norm": float("inf"), "lr": 6e-5, "loss_scale": 2048, "tokens_sec": 990},
]

batches = {
    1: {"id": "batch_001", "max_len": 2048, "valid_label_ratio": 0.82, "has_invalid_value": False, "source": "web_clean"},
    2: {"id": "batch_002", "max_len": 2048, "valid_label_ratio": 0.80, "has_invalid_value": False, "source": "web_clean"},
    3: {"id": "batch_003", "max_len": 2048, "valid_label_ratio": 0.79, "has_invalid_value": False, "source": "code_clean"},
    4: {"id": "batch_004_bad_mask", "max_len": 8192, "valid_label_ratio": 0.03, "has_invalid_value": False, "source": "sft_mixed"},
    5: {"id": "batch_005", "max_len": 2048, "valid_label_ratio": 0.78, "has_invalid_value": False, "source": "web_clean"},
    6: {"id": "batch_006_invalid", "max_len": 2048, "valid_label_ratio": 0.81, "has_invalid_value": True, "source": "math_synth"},
}

rank_losses = {4: [5.8, 5.7, 5.9, 5.8], 6: [2.1, float("nan"), 2.0, 2.1]}

checkpoint = {
    "model": True,
    "optimizer": True,
    "scheduler": False,
    "scaler": False,
    "random_state": True,
    "data_position": False,
}

seq_limit = 4096
valid_label_min = 0.2
grad_threshold = 5.0
spike_ratio = 2.0

def is_finite(x):
    return math.isfinite(x)

def finite_losses_before(step):
    known_spikes = {candidate_step for candidate_step, _ in spike_steps}
    return [
        r["loss"]
        for r in history
        if r["step"] < step and r["step"] not in known_spikes and is_finite(r["loss"])
    ]

spike_steps = []
nonfinite_steps = []
grad_explosion_steps = []
bad_batches = []
rank_mismatch_steps = []

for record in history:
    step = record["step"]
    loss = record["loss"]
    grad_norm = record["grad_norm"]

    if not is_finite(loss) or not is_finite(grad_norm):
        nonfinite_steps.append(step)

    previous = finite_losses_before(step)
    if previous and is_finite(loss):
        baseline = median(previous[-3:])
        if loss / baseline >= spike_ratio:
            spike_steps.append((step, round(loss / baseline, 2)))

    if is_finite(grad_norm) and grad_norm > grad_threshold:
        grad_explosion_steps.append((step, grad_norm))

    batch = batches[step]
    reasons = []
    if batch["max_len"] > seq_limit:
        reasons.append("too_long")
    if batch["valid_label_ratio"] < valid_label_min:
        reasons.append("low_valid_label_ratio")
    if batch["has_invalid_value"]:
        reasons.append("invalid_value")
    if reasons:
        bad_batches.append((step, batch["id"], reasons))

for step, losses in rank_losses.items():
    finite = [x for x in losses if is_finite(x)]
    has_nonfinite = len(finite) != len(losses)
    spread = max(finite) - min(finite) if finite else float("inf")
    if has_nonfinite or spread > 0.5:
        rank_mismatch_steps.append((step, has_nonfinite, round(spread, 3) if is_finite(spread) else "inf"))

missing_checkpoint = [name for name, ok in checkpoint.items() if not ok]
healthy_steps = [
    r["step"]
    for r in history
    if r["step"] not in nonfinite_steps and r["step"] not in [s for s, _ in spike_steps]
]
rollback_step = max(step for step in healthy_steps if step < min(nonfinite_steps or [10**9]))

recommendations = []
if bad_batches:
    recommendations.append("quarantine_bad_batches")
if grad_explosion_steps:
    recommendations.append("lower_lr_or_enable_clipping")
if nonfinite_steps:
    recommendations.append("switch_bf16_or_check_loss_scaling")
if missing_checkpoint:
    recommendations.append("inspect_checkpoint_continuity")
if rank_mismatch_steps:
    recommendations.append("compare_rank_batches_and_collectives")

diagnostics = {
    "spike_detected": spike_steps == [(4, 2.66)],
    "nonfinite_detected": nonfinite_steps == [6],
    "bad_batch_detected": bad_batches[0][1] == "batch_004_bad_mask",
    "checkpoint_gap_detected": missing_checkpoint == ["scheduler", "scaler", "data_position"],
    "rank_mismatch_detected": rank_mismatch_steps == [(6, True, 0.1)],
    "rollback_before_nonfinite": rollback_step == 5,
}

print("spike_steps=", spike_steps)
print("nonfinite_steps=", nonfinite_steps)
print("grad_explosion_steps=", grad_explosion_steps)
print("bad_batches=", bad_batches)
print("rank_mismatch_steps=", rank_mismatch_steps)
print("missing_checkpoint=", missing_checkpoint)
print("rollback_step=", rollback_step)
print("recommendations=", recommendations)
print("diagnostics=", diagnostics)
assert all(diagnostics.values())
```

期望输出：

```text
spike_steps= [(4, 2.66)]
nonfinite_steps= [6]
grad_explosion_steps= [(4, 7.5)]
bad_batches= [(4, 'batch_004_bad_mask', ['too_long', 'low_valid_label_ratio']), (6, 'batch_006_invalid', ['invalid_value'])]
rank_mismatch_steps= [(6, True, 0.1)]
missing_checkpoint= ['scheduler', 'scaler', 'data_position']
rollback_step= 5
recommendations= ['quarantine_bad_batches', 'lower_lr_or_enable_clipping', 'switch_bf16_or_check_loss_scaling', 'inspect_checkpoint_continuity', 'compare_rank_batches_and_collectives']
diagnostics= {'spike_detected': True, 'nonfinite_detected': True, 'bad_batch_detected': True, 'checkpoint_gap_detected': True, 'rank_mismatch_detected': True, 'rollback_before_nonfinite': True}
```

这个 demo 的重点不是模拟真实训练，而是把训练 debug 的日志口径固定下来：先定位 step，再关联 batch、grad、rank 和 checkpoint。真实项目里可以把这些检查接到 Trainer callback、训练日志、数据样本落盘和告警系统里。

## 13. 把一次排查写成可复盘记录

一次高质量的 debug 不应只留下“把 lr 调小后恢复了”这句话，而应留下别人可以重放和质疑的证据。最小记录可以包含四个维度：

| 维度 | 记录内容 | 解释作用 |
| --- | --- | --- |
| 现象 | spike、非有限值、OOM、hang 或吞吐下降 | 明确要解释的症状，而不是笼统写“不稳定” |
| 时间 | 第一次出现的 global step、micro-step 和阶段 | 区分 forward、backward、更新和通信 |
| 输入与状态 | batch id、decode 文本、label mask、lr、grad norm、loss scale、rank | 判断数据、目标、优化器和精度哪一层先异常 |
| 对照与变更 | 单卡结果、健康 checkpoint、最近一次代码/数据/超参改动 | 判断现象是否可复现，以及修复是否真正对应假设 |

把记录按相同口径保存，下一次遇到相似问题时才可以比较。它也能防止一个常见误区：修复后 loss 恢复，并不等于模型质量、验证集表现、吞吐和 checkpoint 连续性都恢复了。

## 14. 把工程判断说清楚：常见面试场景

面试题的价值不在于让候选人背出某个阈值，而在于观察他能否把现象、证据、假设、实验和回归验证连起来。下面的回答示例沿用本章的日志口径；实际工作中还要根据模型规模、并行方式和精度配置补充细节。

### 问法 1：训练 loss 突然 spike，你怎么排查？

可以这样答：

```text
我会先定位 spike 第一次出现的 step，然后看该 step 的 batch、loss、grad norm、learning rate 和 mixed precision 状态。先检查是否是坏数据或 label mask 错误，再看 warmup 是否太短、peak lr 是否过大、grad clipping 是否生效。如果出现 NaN/Inf，还要检查 FP16 overflow 和 optimizer state。必要时从 spike 前 checkpoint 恢复，并做单变量实验确认原因。
```

### 问法 2：NaN 怎么定位？

可以这样答：

```text
NaN 要定位第一个坏掉的 step 和第一个坏掉的张量。我会检查 loss、logits、grad norm、参数范数和 optimizer state，确认是 forward 就 NaN，还是 backward 后梯度 NaN。然后检查 batch、学习率、mixed precision、loss scaling 和 gradient clipping。分布式场景还要看是不是某个 rank 先 NaN。
```

### 问法 3：为什么 train loss 下降但模型效果很差？

可以这样答：

```text
可能是数据重复、评估污染、训练目标和评估目标不一致、label mask 错误或模型只学到了格式。我要看 validation loss、per-domain loss、去重和污染检测，也会 decode 样本检查模型到底在学哪些 token。不能只看 train loss 判断模型能力。
```

### 问法 4：分布式训练 hang 住怎么办？

可以这样答：

```text
先看是否某个 rank 提前报错或 OOM，其他 rank 在 collective 通信处等待。然后检查所有 rank 的 step、batch size、dataloader 长度和 collective 调用顺序是否一致。排查时先单卡，再单机多卡，最后多机，逐步缩小问题范围。
```

### 问法 5：恢复 checkpoint 后 loss 突然异常怎么办？

可以这样答：

```text
我会检查是否完整恢复了 model、optimizer、scheduler、scaler、random state 和数据位置。如果只恢复模型权重，AdamW 动量和学习率 schedule 不连续，loss 可能跳变。还要确认 tokenizer、config、并行切分和 checkpoint 格式一致。
```

## 15. 本章小结

本章核心结论：

1. 训练 debug 要先定位层级，再修复症状。
2. Loss spike 可能来自数据、学习率、梯度、精度或恢复状态。
3. NaN/Inf 要定位第一个坏 step 和第一个坏张量。
4. 梯度异常要结合 grad norm、loss mask 和 optimizer 参数分组看。
5. 数据 batch 和 label mask 是最常见、也最容易被忽略的根因。
6. 学习率过大、warmup 不足和 gradient clipping 缺失会导致训练不稳定。
7. 混合精度问题要关注 FP16 overflow、loss scaling 和高风险 kernel。
8. 分布式问题要先缩小到单卡/单机，再排查通信和 rank 状态。
9. Checkpoint 恢复必须包含 optimizer、scheduler、scaler、随机状态和数据位置。
10. 面试中要用结构化证据链回答训练 debug，而不是只说“调参”。

训练稳定性最终不是“把曲线调得好看”，而是能说明每一次参数更新使用了什么数据、什么监督目标、什么数值格式和什么优化器状态；出现异常时，能够定位第一个坏掉的阶段，保存可复现的现场，并用对照实验验证修复没有改变问题定义。

## 16. 资料与进一步阅读

本章涉及的框架行为和机制可从以下原始资料继续核对：

- [PyTorch autograd anomaly detection](https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly)：说明 `detect_anomaly` 如何记录反向图信息，以及 `check_nan` 对反向异常的调试作用；它适合定位问题，不适合常驻生产训练。
- [PyTorch `clip_grad_norm_`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)：说明 global norm clipping 和 `error_if_nonfinite` 的 API 语义。
- [PyTorch automatic mixed precision](https://pytorch.org/docs/stable/amp.html)：说明 autocast、`GradScaler`、unscale 和跳过非有限更新的行为。
- [PyTorch DistributedDataParallel notes](https://pytorch.org/docs/stable/notes/ddp.html)：说明 DDP 的梯度同步模型和各 rank 参与 collective 的基本约束。
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)：说明分布式 checkpoint 的保存与加载接口边界。
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)：Pascanu 等人讨论梯度爆炸与梯度裁剪；论文解释机制，不给出适用于所有 LLM 的固定 clipping threshold。
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)：Micikevicius 等人说明 FP16 训练中的 loss scaling、FP32 master weights 和混合精度计算。

官方文档适合核对框架的实际行为，原始论文适合核对机制与实验背景；`tau_spike`、`grad_threshold`、warmup 长度和“跳过多少异常样本”等数值仍需在目标数据、模型和硬件上重新验证。
