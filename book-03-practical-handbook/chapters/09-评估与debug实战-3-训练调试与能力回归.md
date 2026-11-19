# 第 9 章（第三部分）：训练调试与能力回归

模型输出是评估对象，训练过程则是它形成的机制。遇到 loss 不降、梯度异常或更新后预测不变，直接调大学习率常会把问题掩盖得更深。本节从数据和标签的不变量开始，逐步检查目标、梯度、参数更新和数值精度，并用小数据过拟合验证训练链条。

## 9.4 排查训练不收敛：先提出能被实验推翻的解释

### 9.4.1 “loss 不降”不是一个足够精确的症状

训练日志出现一条水平线，可能意味着标签错了、数据没有变化、学习率太小、梯度被截断、参数没有加入优化器、损失计算错了、padding 处理错了，也可能是模型已经在当前数据上达到极限。相反，loss 下降也不代表训练正确：模型可能只记住了重复样本，验证集可能同步泄漏，生成阶段的 chat template 还可能不匹配。

排查时，先把“loss 不降”这类模糊症状拆成几个能用实验检验的原因。每次只改一个主要变量，并保留最小复现样本；如果同时改好几处，就无法判断是哪项改动起了作用。

### 先问模型能不能学会一小批样本

如果一个模型连 8--32 条固定样本都无法过拟合，直接扩大到数百万条数据通常只会把问题隐藏在更长的日志里。先检查输入、标签、损失和梯度，再讨论更大的模型和更复杂的学习率策略。

一个最小训练实验应固定：

```text
样本文件和顺序
tokenizer 与 padding 规则
模型初始化种子
batch size 与梯度累积
学习率、warmup、weight decay
精度和梯度裁剪
损失 mask
保存与恢复方式
```

每个条件都能写进实验记录，而不是只写“训练 3 个 epoch”。

### 把 loss 拆回目标、梯度和参数更新

自回归语言模型的 token-level 交叉熵可以写成：

```math
\mathcal{L}(\theta)=
-\frac{1}{|T|}\sum_{t\in T}\log p_\theta(y_{t}\mid x_{1:t-1})
```

集合 $T$ 只应包含有效标签位置，并且必须满足 $|T|>0$；没有有效标签时，平均损失未定义，训练循环应报告数据或 mask 配置错误，而不是返回 0。padding、用户输入、被屏蔽的 assistant 前缀是否进入 $T$，会直接改变训练目标。

参数更新的抽象形式是：

```math
\theta_{k+1}=\theta_{k}-\eta_{k}\,u_{k}(g_{k}),\qquad g_{k}=\nabla_\theta\mathcal{L}_{k}
```

其中 $u_{k}$ 表示优化器对梯度的变换。除了 loss，还应记录梯度范数、参数范数、更新范数和有效学习率：

```math
G_{k}=\lVert g_{k}\rVert_{2},\qquad
P_{k}=\lVert\theta_{k}\rVert_{2},\qquad
U_{k}=\lVert\theta_{k+1}-\theta_{k}\rVert_{2},\qquad
q_{k}=\frac{U_{k}}{P_{k}},\quad P_{k}>0
```

这里的更新量比 $q_{k}$ 只在 $P_{k}>0$ 时定义；若参数范数为 0，应单独报告“相对更新量不适用”，不要用一个任意的 $\varepsilon$ 把定义域问题隐藏起来。当 $G_{k}$ 接近 0，可能是梯度确实很小，也可能是 loss 没有连接到目标参数；当 $G_{k}$ 极大或出现 NaN，常见原因是学习率、数据、精度、非法 logits 或 loss mask；当 $G_{k}$ 正常而 $U_{k}$ 接近 0，可能是学习率太小、优化器状态异常或参数未被更新。

### 9.4.2 第一层：数据和标签不变量

在看 GPU 利用率之前，先对一个 batch 做不变量检查：

```text
输入长度是否在预期范围
标签是否落在词表范围内
有效 label 数是否大于零
输入与标签是否按一个 token 对齐
padding 位置是否使用 ignore index
样本与标签是否在 shuffle 后仍然配对
同一个 batch 是否每次读取相同版本
```

对于监督微调，最常见的隐蔽错误是 assistant 标签全部被 mask，导致有效 token 数为 0；或者把输入和目标错位，模型在学习预测错误位置。打印一条样本的 token、label、mask 和解码文本，往往比看几百行训练日志更快。

### 9.4.3 第二层：在极小数据上过拟合

选择少量样本，关闭复杂增强和随机采样，固定 batch，观察训练 loss 是否明显下降并且生成结果是否接近训练目标。这个实验不是为了获得泛化能力，而是验证计算图、数据通路和优化器确实连接起来。

若小数据不能过拟合，按下面顺序缩小范围：

1. 用一个已知可学习的简单函数或分类任务验证训练循环。
2. 检查参数 requires_grad、optimizer 参数列表和梯度是否为 None。
3. 暂时关闭混合精度、梯度累积、梯度检查点和复杂 scheduler。
4. 把学习率提高或降低一个数量级做敏感性对照，而不是连续微调几个百分点。
5. 检查 loss 的分母是否把有效 token 数算成了总 padding 长度。

小数据过拟合成功后，再逐项恢复真实设置；每恢复一项都运行相同的 smoke test。

### 9.4.4 第三层：损失、梯度和数值精度

常见训练曲线与可能原因可以这样读：

```text
loss 从第一步就是 NaN：非法输入、log(0)、溢出、精度或初始化问题
loss 先剧烈上升再 NaN：学习率过大、梯度爆炸、异常 batch
loss 几乎水平且梯度很小：学习率太小、目标被 mask、参数未更新
loss 大幅振荡：学习率过大、batch 太小、数据分布混合或梯度噪声大
训练 loss 降、验证 loss 升：过拟合、切分泄漏反转、训练与验证模板不一致
训练 loss 正常、生成全是重复：解码、EOS、mask 或 tokenizer 配置问题
```

梯度裁剪可以限制范数。对裁剪上限 $c\ge0$，当梯度范数大于 0 时：

```math
g\leftarrow g\cdot\min\left(1,\frac{c}{\lVert g\rVert_{2}}\right),\quad \lVert g\rVert_{2}>0
```

当 $\lVert g\rVert_{2}=0$ 时梯度保持为 0；实现中若用 $\varepsilon>0$，它只是防止浮点除零的数值稳定项，不应被当作解决梯度为空的办法。裁剪可以缓解异常梯度，但不会修复错误标签和错误损失。若“触发裁剪的步数 / 总步数”长期接近 1，说明训练经常超过上限；若缩放系数 $\min(1,c/\lVert g\rVert_{2})$ 接近 1，则表示裁剪很轻或未裁剪。应区分这两个量，再检查学习率、数据和数值稳定性。

Adam 的基本形式是：

```math
m_{t}=\beta_{1}m_{t-1}+(1-\beta_{1})g_{t},\qquad
v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})g_{t}^2
```

```math
\hat m_{t}=\frac{m_{t}}{1-\beta_{1}^t},\qquad
\hat v_{t}=\frac{v_{t}}{1-\beta_{2}^t},\qquad
\theta_{t}=\theta_{t-1}-\eta\frac{\hat m_{t}}{\sqrt{\hat v_{t}}+\epsilon}
```

Adam 式中的 $\epsilon>0$ 是数值稳定项；同时通常要求 $0\le\beta_{1},\beta_{2}<1$，学习率 $\eta>0$，并保证分母有限。恢复 checkpoint 时若漏掉 optimizer state，虽然模型参数被加载，动量和二阶统计却从头开始，前几步的行为可能明显不同。恢复训练的日志必须记录 global step、scheduler step、optimizer state 和随机状态。

### 9.4.5 一个零依赖的最小训练实验

下面用 Python 标准库训练一个二分类线性模型。它不代表 Transformer 的训练机制，却可以验证数据、梯度、学习率和损失曲线的基本关系；如果连这个最小循环都不能稳定下降，就不应先怀疑复杂模型的“能力不足”。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import math


DATA = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 0.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 1.0),
]


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


weights = [0.0, 0.0]
bias = 0.0
learning_rate = 1.0


def validate_data(data):
    if not isinstance(data, list) or not data:
        raise ValueError("DATA must contain at least one sample")
    for features, target in data:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError("each feature vector must contain two values")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) for value in features):
            raise ValueError("features must be finite numbers")
        if (not isinstance(target, (int, float)) or isinstance(target, bool)
                or not math.isfinite(target) or target not in (0.0, 1.0)):
            raise ValueError("binary target must be 0.0 or 1.0")


if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
    raise TypeError("learning_rate must be numeric")
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise ValueError("learning_rate must be finite and positive")
validate_data(DATA)

for step in range(300):
    grad_w = [0.0, 0.0]
    grad_b = 0.0
    loss = 0.0
    for features, target in DATA:
        logit = sum(w * x for w, x in zip(weights, features)) + bias
        probability = sigmoid(logit)
        loss += max(logit, 0.0) - target * logit + math.log1p(math.exp(-abs(logit)))
        error = probability - target
        for index, feature in enumerate(features):
            grad_w[index] += error * feature
        grad_b += error
    scale = 1.0 / len(DATA)
    if not math.isfinite(loss) or not all(math.isfinite(value) for value in grad_w + [grad_b]):
        raise FloatingPointError("loss or gradient is not finite")
    weights = [
        weight - learning_rate * scale * gradient
        for weight, gradient in zip(weights, grad_w)
    ]
    bias -= learning_rate * scale * grad_b
    if step in {0, 1, 10, 50, 299}:
        print(step, round(loss * scale, 4), [round(x, 3) for x in weights], round(bias, 3))

predictions = []
for features, _ in DATA:
    logit = sum(w * x for w, x in zip(weights, features)) + bias
    predictions.append(int(sigmoid(logit) >= 0.5))
print("predictions=", predictions)
```
损失采用 logits 形式的稳定二元交叉熵，避免截断概率后让损失与梯度失配。首步可手算：初始 logit 都是 0，概率都是 0.5，平均损失为 $\ln 2\approx0.6931$；第一维梯度合计 $-0.5-0.5=-1$，除以 4 后为 $-0.25$，第二维梯度合计 $0.5-0.5=0$，bias 梯度也为 0。学习率为 1 时，首步更新为 $w=[0.25,0]$、 $b=0$，四条新 logit 为 $[0,0,0.25,0.25]$，对应更新后平均损失约为 0.6345。代码日志里的 loss 属于本步更新前的参数，而权重和 bias 属于更新后；最终预测单独使用最后一次更新后的参数。当前标签由第一维特征决定，数据线性可分，训练目标应趋向预测 `[0, 0, 1, 1]`。把 learning_rate 改成非常大的数，可以观察振荡或数值异常；把 DATA 中的 target 全改成同一个值，则可以观察到模型只学到常数基线。这个实验把“训练失败”转成了可以人为制造和观察的现象。

### 9.4.6 真实训练中的定位顺序

当大型训练出现异常，可以按成本从低到高推进：

```text
步骤 1：保存异常 batch、输入解码和有效 label 数
步骤 2：检查 loss 是否有限、梯度是否存在、参数是否变化
步骤 3：在 8--32 条样本上关闭复杂优化组件做过拟合
步骤 4：单独测试 tokenizer、collator、mask 和模型 forward
步骤 5：做学习率、精度、clip 的大范围对照
步骤 6：恢复混合精度、并行、累积和 checkpoint，逐项回归
步骤 7：扩大数据并检查验证集、生成质量和资源指标
```

每一步都应留下“假设—改动—观测—结论”。例如：

```text
假设：所有标签被 mask，导致 loss 没有有效项。
改动：打印 valid_label_count，并让一个样本只保留 assistant token。
观测：原配置为 0，修正后为 42，loss 开始下降。
结论：根因在 collator，不是学习率；恢复原学习率继续验证。
```

这种记录比“调了很多参数后终于能训”更有价值，因为它能指导下一次相似故障。

小样本过拟合测试提供一个有用的分岔点：若有效 label 数为零，先修监督 mask；若 label 存在而 loss 完全不变，检查 logits、梯度和 optimizer 参数组是否连接；若几条样本可以过拟合、扩大后才发散，再看学习率、梯度裁剪、精度和 batch 构成。反过来，能记住 8 条样本不等于训练方案有效，它只说明模型、损失和更新至少在这条最短路径上能够工作；泛化仍要靠独立切分和任务指标判断。

### 9.4.7 训练稳定性与泛化不是同一件事

训练 loss 下降只说明优化目标在当前数据上变小。还应检查：

- 验证 loss 和按类别的验证分数；
- 生成的格式、长度、EOS 和重复率；
- 训练/验证数据的时间、实体和模板分布；
- 长短输入、不同语言和安全样本的分组结果；
- 峰值显存、吞吐、梯度溢出和 checkpoint 可恢复性。

如果训练集和验证集高度近重复，验证 loss 也可能很漂亮；如果训练目标只监督答案而线上要求工具调用，loss 下降也不保证协议正确。训练调试还要和[第一部分的评测协议](09-评估与debug实战-1-评测协议与benchmark.md)及[第二部分的数据污染检查](09-评估与debug实战-2-数据与事实性核验.md)一起看。

### 9.4.8 资料与边界

[PyTorch 2.15 Autograd 文档](https://docs.pytorch.org/docs/2.15/notes/autograd.html)说明了梯度记录和计算图行为；[Adam 论文](https://arxiv.org/abs/1412.6980)给出了自适应优化器的基本形式；[Deep Learning](https://www.deeplearningbook.org/)系统讨论了优化、数值和泛化问题。这里的调试顺序是工程方法，不是保证所有训练故障都能一次定位的算法；复杂分布式训练还需要结合通信、内存、数据加载和硬件错误日志。

训练链条能正常更新参数后，还要确认更新改变了哪些能力。接下来读微调后的能力回归，比较目标任务收益和其他任务退化。

训练稳定不代表微调没有代价。一个模型可以更会回答新业务问题，同时更不会遵守原有格式、处理通用问题或遵守安全边界。接下来把目标能力和保留能力分开测量，再比较回放、降低更新强度、输出锚定、参数正则与任务路由等解决路线。

## 9.5 微调后的能力退化：从目标任务提升到整体能力衡量

### 9.5.1 一个模型可以同时变好和变坏

微调完成后，目标任务分数上涨并不意味着模型整体能力上涨。一个客服模型可能更会使用企业术语，却更容易把普通问题拒答；一个代码模型可能更擅长某种仓库风格，却不再遵守 JSON 协议；一个领域模型可能在专业问答上提高，却忘记通用数学、语言或安全边界。

这种现象通常被称为灾难性遗忘，但实际项目中的退化不止一种：

1. 记忆遗忘：新分布的梯度改变了原有参数表示。
2. 任务干扰：新任务与旧任务要求不同的输出策略。
3. 格式回归：模型学会目标内容，却破坏结构化协议、停止符或工具 schema。
4. 安全回归：窄领域数据覆盖了原有拒答、隐私和权限行为。
5. 分布回归：训练数据与线上输入的长度、语言、噪声或多轮形式不同。
6. 解码回归：模型参数变化不大，但最佳温度、停止条件和长度分布发生变化。

因此，评估微调不能只保留一个 target score。至少要同时测量目标任务、通用能力、格式、安全、长短输入、旧版本回归和资源代价。

### 分开看新任务和原有任务

第一张是“它有没有学会新任务”，第二张是“它有没有破坏原来会做的事情”。如果只看第一张，任何过拟合都可能被误认为成功；如果只看第二张，又会错过有价值的领域适配。

把旧能力样本记为 $D_{\mathrm{retain}}$，新任务样本记为 $D_{\mathrm{target}}$，至少报告：

```text
target_score_before -> target_score_after
retain_score_before -> retain_score_after
format_validity_before -> format_validity_after
safety_error_before -> safety_error_after
latency_and_cost_before -> latency_and_cost_after
```

目标任务提升和保留能力下降同时出现时，问题不是“模型到底好还是坏”，而是需要判断业务愿意支付多少退化代价，或者是否应采用路由、混合数据和更小的参数更新。

### 目标任务与保留能力是两类目标

设目标任务损失为 $\mathcal{L}_{\mathrm{target}}$，保留集损失为 $\mathcal{L}_{\mathrm{retain}}$，协议和安全约束分别为 $\mathcal{L}_{\mathrm{format}}$ 与 $\mathcal{L}_{\mathrm{safety}}$，一种带锚定项的抽象目标是：

```math
\mathcal{L}_{\mathrm{total}}=
\mathcal{L}_{\mathrm{target}}
+\lambda_{\mathrm{retain}}\mathcal{L}_{\mathrm{retain}}
+\lambda_{\mathrm{format}}\mathcal{L}_{\mathrm{format}}
+\lambda_{\mathrm{safety}}\mathcal{L}_{\mathrm{safety}}
+\lambda_{\mathrm{anchor}}D\bigl(p_\theta\Vert p_{\theta_{0}}\bigr)
+\Omega(\theta)
```

$\theta_{0}$ 是微调前模型， $D$ 可以是输出分布的 KL 距离， $\Omega$ 可以是参数或 adapter 的正则项。实际训练未必显式使用这一个总目标，但这个分解提醒我们：只优化新任务 loss，就没有机制保护未被采样的旧能力。

目标提升量和保留退化量分别为：

```math
\Delta_{\mathrm{target}}=S_{\mathrm{target}}(\theta)-S_{\mathrm{target}}(\theta_{0})
```

```math
\Delta_{\mathrm{retain}}=S_{\mathrm{retain}}(\theta)-S_{\mathrm{retain}}(\theta_{0})
```

当 $\Delta_{\mathrm{target}}>0$ 而 $\Delta_{\mathrm{retain}}<0$ 时，应报告这两个方向，而不是把它们混成一个平均分。平均分掩盖某类样本退化的方式，与[第一部分讨论的 micro average](09-评估与debug实战-1-评测协议与benchmark.md)风险相同。

### 9.5.2 先排除“评估看起来退化”的假象

微调后分数下降，不一定是遗忘。首先检查以下条件是否保持一致：

```text
模型调用是否使用同一个 chat template
system、user、assistant 角色是否一致
tokenizer 是否改变，特殊 token 是否映射一致
temperature、top_p、max_tokens 是否一致
停止词和 JSON 解析规则是否一致
上下文长度、截断方向和 padding 是否一致
评估数据是否发生时间、权限或文档版本变化
```

例如，模型本身能够生成正确 JSON，但新的 tokenizer 把结束符处理不同，runner 在等待停止符时截断了输出；这应该归入协议或服务回归，而不是直接写成“模型遗忘 JSON”。

### 9.5.3 退化曲线与有效训练预算

训练步数、学习率和数据重复次数共同决定模型被新分布推动多远。设训练 token 数为 $T$，每个样本平均长度为 $\ell$，样本重复或 epoch 数为 $e$，则粗略关系为：

```math
T\approx e\cdot N\cdot\ell
```

当数据集很小而 epoch 很多时，模型在有限表达模式上反复更新，容易提高目标集分数并损失多样性。学习率过大时，单步参数变化更显著；对 LoRA 等参数高效微调，rank、alpha 和目标模块也会改变可表达的更新空间。

可以记录相对更新量：

```math
r_{\mathrm{update}}=\frac{\lVert\theta-\theta_{0}\rVert_{2}}{\lVert\theta_{0}\rVert_{2}},\quad \lVert\theta_{0}\rVert_{2}>0
```

这个比值只在 $\lVert\theta_{0}\rVert_{2}>0$ 时有定义；若基座参数范数为 0，应报告相对更新量不适用，不能用任意 $\varepsilon$ 掩盖分母为 0。它不能单独预测质量，但能帮助比较不同微调方案是否施加了相近强度的改变。对 LoRA 等参数高效微调，还应记录 adapter 参数范数、合并前后行为以及目标层集合。

### 9.5.4 解决退化的几条路线

#### 路线一：混合回放数据

在新任务 batch 中混入旧能力、通用、安全和格式样本，让梯度持续看到需要保留的行为。混合比例不应只按样本数量决定，因为安全和结构化失败的代价可能远高于普通问答。

设新任务 batch 比例为 $\alpha$，回放比例为 $1-\alpha$：

```math
\mathcal{L}_{\mathrm{mix}}=
\alpha\mathcal{L}_{\mathrm{target}}+
(1-\alpha)\mathcal{L}_{\mathrm{replay}}
```

这里 $0\le\alpha\le1$。新任务比例 $\alpha$ 太低，适配可能较慢或不足；太高则对旧能力的保护减弱。上述损失混合等价于按该比例采样，需要两类损失的归一化口径一致；按样本采样但按 token 平均时，还应考虑长度差异。对回放数据要保留版本和去重信息，避免把测试样例重新混入训练。

#### 路线二：降低更新强度

降低学习率、减少 epoch、使用 warmup、冻结更多层或限制 adapter 目标模块，通常能减少大范围漂移。它们可能牺牲目标任务峰值，但有助于保留广泛能力。

#### 路线三：输出分布锚定

在保留样本上让微调模型不要偏离基座模型过远，可使用 KL 或 logits 蒸馏：

```math
\mathcal{L}_{\mathrm{KL}}=
\frac{1}{|T|}\sum_{t\in T}
D_{\mathrm{KL}}\left(
p_{\theta_{0}}(\cdot\mid x_{1:t-1})
\Vert
p_{\theta}(\cdot\mid x_{1:t-1})
\right)
```

这里要求有效位置集合 $T$ 非空、两模型使用同一词表和上下文，且学生分布在教师具有正概率的位置也为正。此处是教师到学生的 KL，前面的总目标只示意一种可能的锚定方向；交换方向会改变优化行为，不能混用。它能约束输出分布，却不自动保证事实、安全或格式；如果基座模型在某个任务上本来就错误，盲目锚定会保留错误。

#### 路线四：参数正则与重要性保护

Elastic Weight Consolidation 使用参数重要性估计，惩罚关键参数偏离旧值：

```math
\mathcal{L}_{\mathrm{EWC}}=
\mathcal{L}_{\mathrm{target}}
+\frac{\lambda}{2}\sum_{i} F_{i}(\theta_{i}-\theta_{0,i})^2
```

$F_{i}$ 常由 Fisher 信息近似。它的直觉是：对旧任务重要的参数不应被新任务随意改变；但重要性估计本身有成本，且不同任务之间可能共享参数，不能保证完全消除干扰。

#### 路线五：路由和分工

如果新任务与通用任务冲突明显，可以使用场景路由、adapter 路由或多个专门模型，而不是要求一个模型在所有目标上用同一组参数达到最优。路由会增加服务复杂度、缓存和评估组合数，但有时比继续扩大单一模型的训练约束更可控。
### 9.5.5 用一个退化矩阵理解方案取舍

假设三种微调方案得到如下结果，分数均归一化到 0--1：

```text
方案 A：目标 0.86，通用 0.80，格式 0.98，安全 0.97，成本 1.00
方案 B：目标 0.91，通用 0.72，格式 0.94，安全 0.95，成本 1.08
方案 C：目标 0.88，通用 0.84，格式 0.99，安全 0.98，成本 1.03
```

方案 B 的目标分最高，却在多个关键样本类别上退步；方案 C 可能更适合产品发布。若只取四个分数的平均，B 与 C 的差距可能被掩盖。正确做法是先定义不可接受的退化条件，再在满足条件的方案中比较收益和成本。

可以用一个简化的效用表示：

```math
U_{\mathrm{deploy}}=w_{t}S_{\mathrm{target}}+w_{g}S_{\mathrm{general}}+w_{f}S_{\mathrm{format}}+w_{s}S_{\mathrm{safety}}-\lambda_{c}(C-C_{0})
```

这里的各项分数应采用相同的 $[0,1]$ 范围， $w_{t},w_{g},w_{f},w_{s}\ge0$ 且权重和为 1； $C$ 是候选方案的成本， $C_{0}$ 是比较基线的成本，两者要用相同单位，且都必须是有限的非负数； $\lambda_{c}\ge0$ 用来把成本差换算到分数尺度。这个公式只用于辅助比较方案，不是训练目标。权重应根据业务风险和用户可能承担的后果来定；安全和权限指标通常不能只靠加权平均决定。

例如使用 §9.5.6 `adapter_large` 的结果，权重为 $(0.40,0.20,0.15,0.25)$，候选与基线成本分别为 1.08 和 1.00、 $\lambda_{c}=0.05$：效用为 $0.40\times0.90+0.20\times0.73+0.15\times0.94+0.25\times0.95-0.05\times(1.08-1.00)=0.8805$。相对基线，它的通用、格式和安全分数分别下降 0.09、0.04 和 0.03；如果发布规则要求这些维度最多下降 0.01，即使效用排序可接受，也必须先按硬门槛拒绝该方案。

### 9.5.6 一个零依赖退化分析器

下面的程序输入基座模型和微调模型在不同样本类别上的结果，计算相对变化、加权效用和表现最差的类别。它不替代人工分析，但能防止只打印一个总平均分。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import json
import math


BASELINE = {
    "target": 0.70,
    "general": 0.82,
    "format": 0.98,
    "safety": 0.98,
}

VARIANTS = {
    "adapter_small": {
        "target": 0.84,
        "general": 0.81,
        "format": 0.98,
        "safety": 0.98,
        "cost": 1.02,
    },
    "adapter_large": {
        "target": 0.90,
        "general": 0.73,
        "format": 0.94,
        "safety": 0.95,
        "cost": 1.08,
    },
    "mixed_replay": {
        "target": 0.87,
        "general": 0.84,
        "format": 0.99,
        "safety": 0.985,
        "cost": 1.04,
    },
}

WEIGHTS = {
    "target": 0.40,
    "general": 0.20,
    "format": 0.15,
    "safety": 0.25,
}


def validate_inputs():
    if not BASELINE or not VARIANTS:
        raise ValueError("baseline and variants must both be non-empty")
    if set(BASELINE) != set(WEIGHTS):
        raise ValueError("baseline metrics and weight metrics must match")
    if not math.isclose(sum(WEIGHTS.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("metric weights must sum to 1")
    for key, weight in WEIGHTS.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid weight for {key}")
    for key, value in BASELINE.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"baseline score for {key} must be in [0, 1]")
    for name, result in VARIANTS.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("variant names must be non-empty strings")
        if set(result) != set(WEIGHTS) | {"cost"}:
            raise ValueError(f"variant {name} has an unexpected metric set")
        for key in WEIGHTS:
            value = result[key]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"variant {name} score for {key} must be in [0, 1]")
        cost = result["cost"]
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"variant {name} cost must be finite and positive")


def analyze(name, result):
    changes = {
        key: round(result[key] - BASELINE[key], 4)
        for key in WEIGHTS
    }
    utility = sum(WEIGHTS[key] * result[key] for key in WEIGHTS)
    utility -= 0.05 * (result["cost"] - 1.0)
    return {
        "name": name,
        "changes": changes,
        "utility": round(utility, 4),
        "worst_change": min(changes.values()),
        "safety_ok": changes["safety"] >= -0.01,
        "format_ok": changes["format"] >= -0.01,
    }


validate_inputs()
report = [analyze(name, result) for name, result in VARIANTS.items()]
report.sort(key=lambda row: row["utility"], reverse=True)
print(json.dumps(report, ensure_ascii=False, indent=2))
```
如果 adapter_large 的加权效用因为目标任务权重较高而排在前面，仍要注意它在通用、格式和安全样本上的表现可能明显下降。代码把变化和约束分开打印，提醒我们“排序最高”不等于“适合使用”。

### 9.5.7 如何区分遗忘、任务干扰和评估协议变化

可以设计四组对照：

```text
同一基座 + 原始评估协议
同一基座 + 新评估协议
微调模型 + 原始评估协议
微调模型 + 新评估协议
```

若两种模型在新协议下都下降，优先检查模板、解析器和解码；若只有微调模型在原始协议下下降，再分析参数更新和数据；若只有某类样本下降，检查训练分布、样本混合和任务冲突；若所有类别都下降，检查 checkpoint、tokenizer 和服务加载。

再配合行为探针：

```text
同义改写是否保持答案
不同长度是否保持格式
添加无关上下文后是否改变事实
需要拒答时是否仍然拒答
工具未执行时是否声称完成
旧任务与新任务交替输入时是否互相污染
```

这些探针不是通用能力的替代品，而是帮助把“分数下降”拆成可解释行为。

### 9.5.8 资料与边界

[EWC 论文](https://arxiv.org/abs/1612.00796)给出了通过参数重要性减轻灾难性遗忘的经典路线；[LoRA 论文](https://arxiv.org/abs/2106.09685)说明了低秩更新如何降低可训练参数量，但参数少不等于没有能力退化；输出锚定、回放和多任务混合的效果依赖数据比例、任务冲突、学习率、目标模块和评估集。

因此，微调方案不能只用“目标分数最高”选择。必须把新能力、旧能力、协议、安全、成本和不同输入类别的结果放在同一份版本化报告里。

能力变化有了分层数据，开放式回答仍需要解释质量差异。接下来读[人工评测与生成质量](09-评估与debug实战-4-人工评测.md)，学习怎样组织人工判断并校准自动裁判。
