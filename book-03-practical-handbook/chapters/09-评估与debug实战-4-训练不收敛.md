# 第 9 章（4）：训练不收敛的系统排查

模型输出是评估对象，训练过程则是它形成的机制。遇到 loss 不降、梯度异常或更新后预测不变，直接调大学习率常会把问题掩盖得更深。这一篇从数据和标签的不变量开始，逐步检查目标、梯度、参数更新和数值精度，并用小数据过拟合验证训练链条。

## 9.4 训练不收敛的系统调试：先建立可证伪的假设

### 9.4.1 “loss 不降”不是一个足够精确的症状

训练日志出现一条水平线，可能意味着标签错了、数据没有变化、学习率太小、梯度被截断、参数没有加入优化器、损失计算错了、padding 处理错了，也可能是模型已经在当前数据上达到极限。相反，loss 下降也不代表训练正确：模型可能只记住了重复样本，验证集可能同步泄漏，生成阶段的 chat template 还可能不匹配。

调试的第一原则是把模糊症状拆成可证伪假设。每次只改变一个主要变量，并保留最小复现样本；否则多个修复同时发生，无法知道哪个真正起作用。

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
-\frac{1}{|T|}\sum_{t\in T}\log p_\theta(y_t\mid x_{1:t-1})
```

集合 $T$ 只应包含有效标签位置，并且必须满足 $|T|>0$；没有有效标签时，平均损失未定义，训练循环应报告数据或 mask 配置错误，而不是返回 0。padding、用户输入、被屏蔽的 assistant 前缀是否进入 $T$，会直接改变训练目标。

参数更新的抽象形式是：

```math
\theta_{k+1}=\theta_k-\eta_k\,u_k(g_k),\qquad g_k=\nabla_\theta\mathcal{L}_k
```

其中 $u_k$ 表示优化器对梯度的变换。除了 loss，还应记录梯度范数、参数范数、更新范数和有效学习率：

```math
G_k=\lVert g_k\rVert_2,\qquad
P_k=\lVert\theta_k\rVert_2,\qquad
U_k=\lVert\theta_{k+1}-\theta_k\rVert_2,\qquad
q_k=\frac{U_k}{P_k},\quad P_k>0
```

这里的更新量比 $q_k$ 只在 $P_k>0$ 时定义；若参数范数为 0，应单独报告“相对更新量不适用”，不要用一个任意的 $\varepsilon$ 把定义域问题隐藏起来。当 $G_k$ 接近 0，可能是梯度确实很小，也可能是 loss 没有连接到目标参数；当 $G_k$ 极大或出现 NaN，常见原因是学习率、数据、精度、非法 logits 或 loss mask；当 $G_k$ 正常而 $U_k$ 接近 0，可能是学习率太小、优化器状态异常或参数未被更新。

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
g\leftarrow g\cdot\min\left(1,\frac{c}{\lVert g\rVert_2}\right),\quad \lVert g\rVert_2>0
```

当 $\lVert g\rVert_2=0$ 时梯度保持为 0；实现中若用 $\varepsilon>0$，它只是防止浮点除零的数值稳定项，不应被当作解决梯度为空的办法。裁剪可以缓解异常梯度，但不会修复错误标签和错误损失。若“触发裁剪的步数 / 总步数”长期接近 1，说明训练经常超过上限；若缩放系数 $\min(1,c/\lVert g\rVert_2)$ 接近 1，则表示裁剪很轻或未裁剪。应区分这两个量，再检查学习率、数据和数值稳定性。

Adam 的基本形式是：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
```

```math
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

Adam 式中的 $\epsilon>0$ 是数值稳定项；同时通常要求 $0\le\beta_1,\beta_2<1$，学习率 $\eta>0$，并保证分母有限。恢复 checkpoint 时若漏掉 optimizer state，虽然模型参数被加载，动量和二阶统计却从头开始，前几步的行为可能明显不同。恢复训练的日志必须记录 global step、scheduler step、optimizer state 和随机状态。

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
损失采用 logits 形式的稳定二元交叉熵，避免截断概率后让损失与梯度失配。首步可手算：初始 logit 都是 0，概率都是 0.5，平均损失为 $\ln 2\approx0.6931$；第一维梯度合计 $-0.5-0.5=-1$，除以 4 后为 $-0.25$，第二维梯度合计 $0.5-0.5=0$，bias 梯度也为 0。学习率为 1 时，首步更新为 $w=[0.25,0]$、$b=0$，四条新 logit 为 $[0,0,0.25,0.25]$，对应更新后平均损失约为 0.6345。代码日志里的 loss 属于本步更新前的参数，而权重和 bias 属于更新后；最终预测单独使用最后一次更新后的参数。当前标签由第一维特征决定，数据线性可分，训练目标应趋向预测 `[0, 0, 1, 1]`。把 learning_rate 改成非常大的数，可以观察振荡或数值异常；把 DATA 中的 target 全改成同一个值，则可以观察到模型只学到常数基线。这个实验把“训练失败”转成了可以人为制造和观察的现象。

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
- 长短输入、不同语言和安全样本的切片；
- 峰值显存、吞吐、梯度溢出和 checkpoint 可恢复性。

如果训练集和验证集高度近重复，验证 loss 也可能很漂亮；如果训练目标只监督答案而线上要求工具调用，loss 下降也不保证协议正确。训练调试还要和[第一篇的评测协议](09-评估与debug实战-1-评测协议与benchmark.md)及[第二篇的数据污染检查](09-评估与debug实战-2-数据污染与泄漏.md)一起看。

### 9.4.8 资料与边界

[PyTorch 2.15 Autograd 文档](https://docs.pytorch.org/docs/2.15/notes/autograd.html)说明了梯度记录和计算图行为；[Adam 论文](https://arxiv.org/abs/1412.6980)给出了自适应优化器的基本形式；[Deep Learning](https://www.deeplearningbook.org/)系统讨论了优化、数值和泛化问题。这里的调试顺序是工程方法，不是保证所有训练故障都能一次定位的算法；复杂分布式训练还需要结合通信、内存、数据加载和硬件错误日志。

训练链条能正常更新参数后，还要确认更新改变了哪些能力。接下来读[微调后的能力回归](09-评估与debug实战-5-微调能力回归.md)，比较目标任务收益和其他任务退化。
