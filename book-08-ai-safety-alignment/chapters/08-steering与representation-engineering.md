# 第八章：Steering 与 Representation Engineering

控制模型行为并不只有一种旋钮。提示词改变输入，解码策略改变输出分布，微调改变参数，
激活干预改变一次前向计算中的中间状态，而系统权限决定模型即使产生了某种意图也能否
造成现实副作用。Steering 主要研究如何在不完整重训模型的情况下，利用表示或激活方向
改变行为；Representation Engineering 则从高层行为出发，寻找可监控、可干预的内部
表示。

本章把“方向有效”拆成四个问题：它是否可重复地改变目标行为，是否保留正常能力，是否
在新任务和新模板上泛化，以及是否会扩大安全暴露面。只有把这些问题分别回答，steering
才是工程证据，而不是一个看起来有效的 prompt 或单次 demo。

```text
正负行为样本 -> 激活差分 -> steering vector -> 强度扫描 -> 行为变化与副作用评估
```

本章只讨论防御性原理、表示监控、over-refusal 调试和安全评估，不提供禁用安全机制、
削弱拒答机制或规避安全边界的操作流程。涉及双用研究时，只描述抽象任务、评估条件和
控制边界。

读者可以沿着“控制位置—行为变化—副作用—系统后果”的顺序理解本章。后文的公式和
demo 都是 toy-level，用来说明差分方向、强度扫描、目标收益和副作用的记录方式，不是
任何真实模型的上线结论。

## 1. 来龙去脉：从 Prompt 控制到表示层控制

### 1.1 最早的控制方式：Prompt

最自然的模型控制方式是写 prompt。

例如：

1. “请用简洁风格回答。”
2. “请逐步推理。”
3. “请扮演严谨审稿人。”
4. “请只输出 JSON。”

Prompt engineering 的优点是简单、无需训练、无需改模型。

缺点也明显：

1. 不稳定。
2. 容易受上下文干扰。
3. 很难精确控制强度。
4. 对复杂行为控制有限。
5. 容易被 prompt injection 影响。

### 1.2 第二阶段：训练层控制

为了更稳定控制模型行为，人们使用：

1. SFT。
2. RLHF。
3. DPO。
4. RLAIF。
5. Safety tuning。
6. Domain fine-tuning。

训练层控制更稳定，但代价更高。

问题包括：

1. 需要数据。
2. 训练成本高。
3. 容易产生副作用。
4. 很难快速开关某个行为。
5. 很难解释内部发生了什么。

### 1.3 第三阶段：表示层和激活层控制

后来研究者开始问：

```text
如果某种行为在模型内部对应某个方向或特征，我们能不能直接监控或干预这个方向？
```

例如：

1. 诚实 vs 不诚实。
2. 有害 vs 无害。
3. 拒答 vs 不拒答。
4. 幻觉 vs 事实性。
5. 某种写作风格。
6. 某种角色倾向。

这就进入了 steering 和 representation engineering。

### 1.4 方法谱系

可以把控制方法看成一条谱系：

```text
Prompt -> Decoding control -> Fine-tuning -> RLHF/DPO -> Activation steering -> Model editing -> System-level control
```

每种方法控制层级不同。

1. Prompt 控制输入。
2. Decoding 控制采样。
3. Fine-tuning 控制参数。
4. RLHF/DPO 控制偏好行为。
5. Activation steering 控制中间表示。
6. Model editing 修改局部知识或行为。
7. System-level control 控制权限、工具和产品边界。

Steering 不是替代其他方法，而是补充工具。

### 1.5 控制位置决定证据和风险

“控制模型”这个短语太宽。至少要区分控制发生在哪一层：

| 控制位置 | 主要改变 | 优势 | 典型边界 |
| --- | --- | --- | --- |
| 输入 | 模型看到的指令和上下文 | 成本低、易回滚 | 容易被上下文和注入影响 |
| 解码 | 下一 token 的选择分布 | 不改权重、实现简单 | 只能改变输出选择，不能补足缺失能力 |
| 激活/表示 | 单次 forward 的中间状态 | 可开关、适合实验和局部控制 | 需要白盒访问，可能产生分布外状态 |
| 参数 | checkpoint 的长期行为 | 持久、可规模化服务 | 训练成本高，副作用难回滚 |
| 系统 | 工具、权限、沙箱和产品流程 | 能直接限制现实副作用 | 不能解释模型内部，也需正确配置 |

小白可以把这些位置看成从“告诉模型怎么做”到“限制它能做什么”的不同旋钮；专家要
同时记录控制对象、作用范围、回滚方式和失败时的最后责任层。一个 activation vector
即使把输出分数提高，也不能替代服务端授权；一个权限策略即使阻止了副作用，也不能证明
模型没有潜在危险能力。

因此 steering 的评估不能只比较前后文本。至少要并列记录目标行为、正常任务、拒答边界、
工具调用、延迟和成本，并把模型控制效果与系统控制效果分开。这样才能知道收益来自
表示干预，还是来自后处理、采样变化或外部策略。

## 2. 小白例子：开车和方向盘

把模型想象成一辆车。

Prompt 像告诉司机：

```text
请开稳一点。
```

Fine-tuning 像重新训练司机。

RLHF 像给司机长期评分，让他学会更符合乘客偏好。

Steering vector 像在行驶过程中轻微转动方向盘。

System-level control 像道路限速、刹车系统、安全带和交通规则。

只靠一句“请开稳”不够。

只靠方向盘也不够。

真实系统需要多层控制。

## 3. 什么是 Steering

Steering 指在不完整重训模型的情况下，用某种方式引导模型行为朝目标方向变化。

常见 steering 目标：

1. 更诚实。
2. 更少幻觉。
3. 更少有害内容。
4. 更愿意拒绝危险请求。
5. 更符合某种语气或格式。
6. 更保守或更开放。
7. 更强调引用证据。

Steering 的形式包括：

1. Prompt steering。
2. Decoding steering。
3. Activation steering。
4. Feature steering。
5. Representation monitoring。
6. Model editing。

本章重点是表示层和激活层 steering。

## 4. Representation Engineering

### 4.1 定义

Representation Engineering 关注模型内部高层表示。

它不是从单个 neuron 或 circuit 逐底层拆解，而是从高层行为出发，寻找模型内部是否存在可监控、可干预的表示方向。

这就是为什么它常被称为 top-down approach。

Mechanistic interpretability 更像：

```text
从底层组件往上理解机制。
```

Representation engineering 更像：

```text
从高层行为往下寻找表示。
```

### 4.2 它解决什么问题

传统机制可解释性非常精细，但成本高。

如果我们只是想监控或调节某个高层行为，例如 honesty、harmlessness、power-seeking，逐个 circuit 逆向可能太慢。

Representation engineering 试图用更工程化的方式：

1. 构造正负样本。
2. 提取内部激活。
3. 找到行为相关方向。
4. 用这个方向监控或干预模型。

### 4.3 优点

1. 比完整机制逆向更快。
2. 适合高层行为监控。
3. 可以用于 steering。
4. 能和 safety eval 结合。
5. 对大模型可能更容易扩展。

### 4.4 缺点

1. 方向可能不是因果机制本身。
2. 正负样本构造会影响结果。
3. 高层行为可能不是单一线性方向。
4. 干预可能有副作用。
5. 解释粒度不如完整 mechanistic analysis。

### 4.5 表示方向、监控信号和控制输入不是一回事

同一个向量可以有三种不同用途。把它用于分类时，它是一个监控信号；把它加入激活时，
它是控制输入；把它拿来解释某个行为时，它是一个机制假设。三种用途的证明责任不同。

例如某个方向能够区分“有引用”和“无引用”样本，只说明它在当前数据上具有可读出性；
如果加上这个方向后引用率上升，说明它有局部控制效果；只有在干预后的引用真正支持
答案、跨模板泛化且没有增加幻觉时，才可以把它当作更有用的工程方向。即便如此，仍不
能说模型内部只有这一条引用机制。

Representation engineering 常使用高层标签，所以数据构造尤其关键。正负样本必须尽量
只在目标行为上不同，控制长度、词汇、主题、角色、拒答模板和情绪语气等因素。否则差分
方向可能学到“句子更长”“出现某个固定短语”或“来自某个数据集”的方向，而不是目标
行为本身。专家会在训练外的语义改写、跨领域样本和无关任务上验证这一点。

## 5. Steering Vector 的核心直觉

### 5.1 表示空间中的方向

假设模型内部激活空间里存在一些和行为相关的方向。

例如：

```text
更诚实的回答 - 更不诚实的回答 = honesty direction
拒答样本 - 正常回答样本 = refusal direction
事实回答 - 幻觉回答 = factuality direction
```

这不是说行为一定真的只由一个方向决定。

它只是一个实用近似：很多高层行为可能在某些层的表示空间里有可线性分离的成分。

### 5.2 如何构造

高层步骤：

1. 构造正例和负例。
2. 跑模型，记录某层激活。
3. 计算正例平均激活和负例平均激活的差。
4. 得到一个 steering vector。
5. 推理时把这个向量加到某些层或位置。
6. 观察行为是否朝目标方向变化。

这里的“正例”和“负例”必须先定义清楚。它们最好在主题、长度、语言、角色和上下文
结构上尽量匹配，只在目标行为或其可接受表达上有计划地不同；否则差分向量可能学到
数据集或模板的标记。若一侧没有样本、某个层位的激活缺失，均值和方向就没有定义，
实验记录应写成 \(N/A\)，不能用零向量假装“没有行为差异”。同一向量还要明确聚合了
哪个 token 位置、哪种 pooling 和哪些样本切片，因为这些选择会改变它的语义。

### 5.3 小白例子

想象你有很多人的照片。

你取“微笑照片”的平均特征，再减去“不微笑照片”的平均特征。

得到一个“微笑方向”。

如果把这个方向加到某张照片的表示上，可能让照片更像在微笑。

Steering vector 对模型行为做类似事情。

### 5.4 差分方向的三个隐含假设

平均激活差隐含了三个假设。第一，正负样本在目标行为之外足够匹配；第二，行为差异在
所选层的表示空间里近似线性可分；第三，把方向加回正常激活时不会破坏其他重要坐标。
任何一个假设不成立，steering vector 都可能只在原始样本上有效。

最简单的均值差还可能被少数异常样本主导。可以使用分层均值、稳健均值或在协方差加权
后再构造方向，但这会引入新的估计选择。报告应说明正负样本数量、分层方式、token/position
聚合方式和是否做中心化；不能只公布一个向量或一个离线提升数字。

层位选择也不是越深越好。早层通常更接近词形、语法和局部上下文，中层可能包含任务和
角色表示，晚层更接近输出读出；不同模型并没有统一层位定律。工程上可以先做 layer
scan，再用 holdout 固定层位，避免在同一测试集上挑选最佳层和最佳强度。

## 6. Contrastive Activation Addition

### 6.1 来龙去脉

Contrastive Activation Addition 这类方法来自一个直觉：

如果模型在“有某种行为”和“没有某种行为”的样本上，内部激活有稳定差异，那么这个差异可以作为控制方向。

它不需要重新训练整个模型。

只需要在推理时修改激活。

### 6.2 核心流程

高层流程：

```text
positive examples -> activations_pos
negative examples -> activations_neg
steering_vector = mean(activations_pos) - mean(activations_neg)
inference activation = activation + alpha * steering_vector
```

其中 `alpha` 控制 steering 强度。

### 6.3 它解决前人什么问题

相比 prompt：

1. 更直接作用于模型内部表示。
2. 可能更稳定。
3. 可调节强度。

相比 fine-tuning：

1. 不需要更新参数。
2. 可快速开关。
3. 适合实验分析。

相比完整 mechanistic interpretability：

1. 不需要完全理解 circuit。
2. 更偏工程可用。

### 6.4 局限

1. 需要白盒访问或至少能访问激活。
2. 方向选择依赖数据。
3. 不同层、位置、强度效果不同。
4. 可能损伤其他能力。
5. 可能只在某些模型或任务上稳定。
6. 可能被滥用来削弱安全行为。

### 6.5 从方向发现到安全实验

一个防御性实验应把方向发现和方向使用分开。发现阶段可以在离线模型上比较正负样本，
但强度和层位一旦在验证集上选择，就应冻结到独立 holdout。使用阶段要扫描一组有限的
\(\alpha\)，记录目标收益曲线和副作用曲线，而不是只展示让目标指标最高的一个点。

还要设置零向量和随机方向对照。若随机方向也能带来相似提升，可能是解码随机性、提示
模板或评估器偏差；若只在某一固定短语上提升，可能是词形捷径。对于安全 steering，
必须加入正常边界、高风险抽象任务、无关任务、多语言和多轮样本，防止把“更多拒绝”误
认为“更安全”。

在生产系统中，activation steering 还要明确 hook 的时机和作用域：是每层每 token，
还是只在特定状态下插入；是同步请求、异步任务还是内部评估；遇到异常时能否关闭并
回退到未干预模型。没有这些条件，方向的离线效果不能转化为部署结论。

## 7. 拒答方向的双重含义

### 7.1 发现的意义

一些研究观察到，在多个开源 chat model 中，拒答行为可能和 residual stream 中某个方向高度相关。

高层理解：

```text
模型是否拒答，可能部分由内部某类拒答表示调节。
```

这很有价值，因为它帮助我们理解 safety tuning 可能如何改变模型内部行为。

### 7.2 安全价值

它可以帮助：

1. 分析拒答机制。
2. 研究 over-refusal。
3. 监控模型是否进入拒答状态。
4. 理解 jailbreak 为什么成功。
5. 设计更稳健的安全训练。

### 7.3 风险

同样的理解也可能被滥用。

如果某个方向控制拒答，那么恶意方可能尝试削弱它。

因此这类研究有双重用途。

本书只讨论安全含义，不给出禁用拒答或绕过安全机制的操作细节。

### 7.4 机制与边界

“单一方向”不应被过度理解成完整机制。

它可能是拒答行为的一个重要中介表示，但模型拒答还涉及：

1. 输入风险识别。
2. 安全策略表示。
3. 多层信息传播。
4. 输出模板生成。
5. 解码和系统策略。

所以更稳妥的说法是：拒答方向是理解和干预拒答行为的重要线索，不是完整 safety 机制。

### 7.5 如何正确解读拒答方向证据

研究报告应至少区分四个命题：方向能否区分拒答与非拒答，干预方向能否改变拒答概率，
这种改变是否在语义改写和多语言样本上复现，以及它是否改变了真正的安全边界。前两个
命题属于表示可读出性和局部控制，后两个命题才开始涉及泛化与安全后果。

一个模型可能在拒答模板上存在清晰方向，却在风险识别、工具权限和输出过滤上使用其他
机制。相反，方向干预可能只改变措辞，让输出看起来更像拒答，却没有减少工具越权或
不安全的任务完成。因此评估不能把 refusal rate 单独当成安全指标，要同时看 unsafe
compliance、safe completion、over-refusal、工具副作用和人工复核成本。

双用风险也要求控制研究材料的粒度。公开章节可以讨论方向构造、强度扫描、误拒和回归
指标；内部评估可以在隔离副本保存更细的激活和干预记录；生产接口不应允许用户直接
提交任意内部向量。这样既能研究安全机制，又不把控制面变成新的绕过入口。

## 8. Steering 的应用

### 8.1 风格控制

例如：

1. 更简洁。
2. 更正式。
3. 更教学化。
4. 更谨慎。

风险：风格改变可能掩盖事实性问题。

### 8.2 Factuality 控制

目标：减少幻觉，提高承认不确定性的倾向。

风险：可能导致模型过度保守，或在该回答时拒答。

### 8.3 Safety 控制

目标：增强拒绝危险请求、降低有害输出。

风险：可能导致 over-refusal，或损害正常安全教育请求。

### 8.4 Persona 控制

目标：改变角色、语气、专业程度。

风险：persona 可能和安全边界冲突。

### 8.5 Tool-use 控制

目标：更谨慎或更主动地调用工具。

风险：过度调用工具增加成本和安全面；调用不足降低任务完成率。

### 8.6 应用必须绑定任务契约

同一个方向在不同任务上的“好”并不相同。对写作助手，简洁方向可能降低事实完整性；
对研究助手，谨慎方向可能提高引用但增加拒答；对 Agent，工具谨慎可能减少副作用，却
也可能导致任务在需要查询时停滞。因此每个应用都要先写任务契约：允许改变什么、必须
保持什么、哪些动作需要人工批准、失败时如何回退。

以事实性 steering 为例，不能只测答案是否更少幻觉，还要检查引用是否真的支持结论、
模型是否诚实表达未知、正常问题是否被过度拒答，以及多轮追问是否保持一致。以工具
控制为例，要并列记录工具选择准确率、参数正确率、越权尝试率、人工确认率、任务完成率
和单位成功成本。方向控制应该改变模型倾向，最终副作用仍由服务端权限和事务语义负责。

## 9. Steering 的评估

### 9.1 行为指标

看目标行为是否改变：

1. Factuality。
2. Refusal rate。
3. Helpfulness。
4. Harmlessness。
5. Conciseness。
6. Citation accuracy。
7. Tool-call accuracy。

### 9.2 副作用指标

必须看副作用：

1. 通用能力是否下降。
2. 输出是否变得奇怪。
3. 是否增加 hallucination。
4. 是否增加 over-refusal。
5. 是否影响多语言。
6. 是否影响长上下文。

### 9.3 鲁棒性指标

测试：

1. 不同 prompt。
2. 不同任务。
3. 不同领域。
4. 多轮对话。
5. Jailbreak。
6. RAG 和 Agent 场景。

### 9.4 表示指标

检查 steering 是否真的影响目标表示。

例如：

1. 目标方向激活变化。
2. 相关 feature 激活变化。
3. 下游行为变化。
4. 因果干预验证。

### 9.5 关键公式与 steering 评估指标速查

设第 \(\ell\) 层、第 \(p\) 个位置的激活为：

$$
h_{\ell,p}\in \mathbb{R}^{d}
$$

**1. 正负样本激活均值**

设正例集合为 \(P\)，负例集合为 \(N\)，某层聚合后的激活为 \(h_i^\ell\)：

$$
\mu_+^\ell=\frac{1}{|P|}\sum_{i\in P} h_i^\ell
$$

$$
\mu_-^\ell=\frac{1}{|N|}\sum_{i\in N} h_i^\ell
$$

这里要求 \(|P|>0\)、\(|N|>0\)，并且两组激活的维度、层位和聚合规则一致。若正负样本
数量差异很大，还应报告分层后的结果或置信区间；只给出一个总体均值，可能让多数类
主导方向。对配对任务，\(P\) 和 \(N\) 还应说明是否由同一语义样本的两个版本构成，
因为独立抽样和配对抽样支持的因果解释不同。

**2. Steering vector**

最常见的差分方向：

$$
v^\ell=\mu_+^\ell-\mu_-^\ell
$$

归一化后：

$$
\bar v^\ell=\frac{v^\ell}{\|v^\ell\|_2+\epsilon}
$$

这个方向只是一种行为相关表示，不应直接解释为完整机制。这里的 \(\epsilon\) 只能
防止浮点除零；如果 \(\|v^\ell\|_2\) 本身接近零，说明当前数据没有稳定的差分方向，
应报告为 \(N/A\) 并回到样本构造或层位选择，而不是把数值噪声放大成控制方向。不同
层的激活尺度也可能不同，所以 \(\alpha\) 只在固定层位、模型精度和归一化约定下可比。

**3. Activation intervention**

推理时在指定层和位置加入 steering：

$$
\tilde h_{\ell,p}=h_{\ell,p}+\alpha \bar v^\ell
$$

其中 \(\alpha\) 是强度系数。真实实验通常要扫描多个 \(\alpha\)，而不是只试一个值。

**4. 表示投影分数**

可以用投影衡量样本是否更接近目标方向：

$$
s_i=\frac{\langle h_i^\ell,\bar v^\ell\rangle}{\|h_i^\ell\|_2+\epsilon}
$$

干预前后投影变化：

$$
\Delta s_i=s_i(\tilde h)-s_i(h)
$$

当 \(\|h_i^\ell\|_2\) 接近 0 时，投影分数的分母会放大噪声，应记为 \(N/A\)。此外，
投影增大只说明样本更靠近选定方向，不说明目标行为一定改善；必须把它和独立的行为
指标配对，否则可能出现“方向信号变强、答案质量变差”的情况。

**5. 目标行为提升**

设 \(q_i^{base}\) 是未干预时目标行为得分，\(q_i(\alpha)\) 是强度为 \(\alpha\) 时的得分：

$$
U_{target}(\alpha)=\frac{1}{M}\sum_i [q_i(\alpha)-q_i^{base}]
$$

其中 \(q_i\) 必须有固定的评分规则和相同量纲，\(M\) 是实际完成配对评估的样本数，要求
\(M>0\)。如果某类任务没有有效样本，目标收益应记录为 \(N/A\)，不能因为没有测量就写成
零提升。若使用自动评估器，还要在资料中说明其版本、阈值和人工抽检规则。

**6. 副作用下降**

设 \(g_i^{base}\) 是通用质量、helpfulness 或任务成功得分，\(g_i(\alpha)\) 是干预后得分：

$$
D_{side}(\alpha)=\frac{1}{M}\sum_i \max(0,g_i^{base}-g_i(\alpha))
$$

这里假设 \(g_i\) 的方向是“越高越好”，并且基线与干预结果在同一任务切片上配对。若
\(M=0\)，或某个切片的质量分数不可比较，\(D_{side}\) 应为 \(N/A\)。Steering 不能只
看目标行为增强，还要看有没有损伤正常能力。

**7. 误拒增量**

设 \(R_{over}^{base}\) 是正常请求误拒率，\(R_{over}(\alpha)\) 是干预后误拒率：

$$
\Delta R_{over}(\alpha)=R_{over}(\alpha)-R_{over}^{base}
$$

安全 steering 很容易把模型推向过度保守，因此这个指标很重要。
其中每个误拒率都必须给出正常请求的分母；正常请求为空时，增量没有定义。还要把
“模型拒答”与“策略服务拦截”分开统计，否则不同控制层的变化会被混在一个比例里。

**8. 安全收益**

设 \(R_{unsafe}^{base}\) 是高风险请求漏拒率，\(R_{unsafe}(\alpha)\) 是干预后漏拒率：

$$
U_{safe}(\alpha)=R_{unsafe}^{base}-R_{unsafe}(\alpha)
$$

这里 \(R_{unsafe}\) 表示高风险样本中的漏拒或不安全完成比例，分母应固定为同一批
预先定义的高风险样本。安全收益为正并不等于风险为零，也不代表安全样本的帮助性没有
下降；样本为空、标签未完成或评估口径变化时应标为 \(N/A\)。

**9. 强度选择**

可以把目标收益和副作用写成一个简单选择问题：

$$
\alpha^*=\arg\max_{\alpha\in A}
[U_{target}(\alpha)+\lambda_s U_{safe}(\alpha)-\lambda_d D_{side}(\alpha)-\lambda_o \Delta R_{over}(\alpha)]
$$

这里的权重应由产品、安全和评估目标事先确定，且各项最好先归一化到可比较的量纲；
不能只按离线分数临时拍。这个目标函数只是选择候选强度的记录工具，不是普遍适用的
安全定律。若某个高风险信号未定义，不能让优化器把它当作零损失继续选择强度。

**10. 把 steering 结果记录成证据向量**

目标收益、安全收益、副作用、误拒和对抗边界回答的是不同问题，不宜合并成一个总布尔
值。对强度 \(\alpha\)，可以记录：

$$
\mathcal{E}_{steer}(\alpha)=
(U_{target},U_{safe},D_{side},\Delta R_{over},R_{jail},\Delta latency,\Delta cost)
$$

其中 \(R_{jail}\) 表示在预先定义的对抗安全集上的边界失败率，\(\Delta latency\) 和
\(\Delta cost\) 分别表示相对未干预系统的延迟和单位成本变化。每个信号都应绑定自己的
分母、样本切片、阈值和后续动作：

1. \(U_{target}\) 太小，说明方向没有稳定达到目标，应回到样本构造或层位选择。
2. \(U_{safe}\) 太小而 \(D_{side}\) 增大，说明控制带来帮助性损失，却没有安全收益。
3. \(\Delta R_{over}\) 增大，说明需要补充正常边界和安全教育样本。
4. \(R_{jail}\) 增大，说明高风险动作必须保持受限并重新做红队评估。
5. 延迟或成本超出任务预算，说明应改为离线分析、低频监控或按场景路由。

即使所有信号暂时落在研究者设定的范围内，结论也只能是“在当前模型、强度、任务分布
和系统控制下继续验证”。它不说明向量完全可靠，更不说明可以绕过服务端权限、人工确认
或回滚机制。

## 10. Steering 和其他方法的关系

### 10.1 和 Prompt Engineering

Prompt engineering 更容易用，但更软。

Steering 更直接，但需要模型内部访问。

### 10.2 和 Fine-tuning

Fine-tuning 改参数，steering 改推理时激活。

Fine-tuning 更持久，steering 更可开关。

### 10.3 和 RLHF / DPO

RLHF/DPO 通过偏好数据改变整体行为。

Steering 可以做局部行为控制或研究工具。

### 10.4 和 Mechanistic Interpretability

Mechanistic interpretability 试图理解机制。

Steering 试图利用表示控制行为。

二者互相促进：理解帮助控制，控制实验帮助验证理解。

### 10.5 和 Model Editing

Model editing 通常修改参数或知识。

Steering 通常在推理时修改激活。

Editing 更持久，steering 更临时。

### 10.6 组合使用时要保留因果边界

一个系统可能同时使用 system prompt、DPO checkpoint、activation steering、输出分类器
和工具权限。上线后观察到行为变化，不能自然地归因于其中某一层。比较 steering 版本
时，至少要保留未干预 baseline、相同解码参数、相同策略服务和相同工具权限；如果同时
更新 checkpoint 或 prompt，应把它们作为新的实验因素。

对安全场景，组合控制还可能互相抵消。训练让模型更愿意说明不确定性，steering 却把
回答推向过度保守；输出过滤器降低了危险文本，但工具服务仍接受了错误参数。工程报告
需要分别记录模型文本、策略判断、工具请求和实际副作用，不能用最终回答“看起来安全”
替代系统 trace。

## 11. 真实项目中的使用方式

### 11.1 研究和调试

Steering 很适合用于研究：

1. 某种行为是否有内部方向。
2. 增强或减弱该方向会发生什么。
3. 该行为和哪些能力有 trade-off。
4. 安全训练是否改变了表示。

### 11.2 安全监控

可以把表示方向作为监控信号之一。

例如：

1. 检测模型是否进入高风险回答状态。
2. 检测拒答状态。
3. 检测不确定性状态。
4. 检测工具调用风险状态。

但不能只靠它做最终决策。

### 11.3 产品控制

实际产品中，steering 可以作为辅助层。

例如在不同模式下调节：

1. 更保守。
2. 更简洁。
3. 更教学。
4. 更严格引用。

但上线前必须做系统评估。

### 11.4 运行时契约

如果 steering 进入服务，运行时必须知道它作用于哪个模型 artifact、哪一层、哪些 token
位置、哪些请求类型和什么强度范围。请求 trace 应记录 steering profile、是否命中、向量
版本、强度、耗时、回退原因和最终策略结果。向量文件本身属于模型控制 artifact，应和
checkpoint、tokenizer、推理引擎及配置一起版本化。

最小的运行时契约还要包含三件事。第一，关闭 steering 后请求能否继续完成；第二，激活
hook 或向量维度不匹配时是否 fail closed 或回退到明确的未干预路径；第三，发生异常时
是否已经产生不可逆工具副作用。对只改文本的同步请求，回退通常比较简单；对长周期
Agent，必须在每次工具提交前重新检查 profile 和权限，不能让一次成功的 steering 请求
永久影响后续状态。

### 11.5 评估窗口与版本迁移

方向效果可能随着模型 checkpoint、量化精度、batching、KV cache、prompt template 和
语言变化。模型升级时，不能只比较最终平均分；应做配对样本的目标收益、副作用、延迟、
工具越权和误拒差异，并重新确认 hook 位置对应相同语义的张量。

若新模型没有稳定复现旧方向，最保守的处理是把它视为新控制 profile，重新构造正负样本
和 holdout，而不是把旧向量强行投影到新模型。迁移成本本身也是 steering 方案的工程
成本，应计入单位成功任务成本和维护预算。

## 12. 风险和局限

### 12.1 可解释性风险

一个方向有效，不代表我们完整理解了机制。

### 12.2 泛化风险

在一个数据集上有效，不代表跨任务有效。

### 12.3 副作用风险

控制一个行为可能影响其他行为。

### 12.4 安全双用风险

能增强安全，也可能被用于削弱安全。

### 12.5 工程复杂度

需要访问内部激活，部署复杂度高于 prompt。

### 12.6 评估难度

需要同时评估目标行为、副作用、鲁棒性和安全边界。

### 12.7 数据构造和评估器偏差

steering vector 直接从正负样本学习，因此样本标签、模板和评估器会进入控制方向。若
正例全部由一种语气生成，方向可能学到语气；若目标收益由同一个 LLM judge 评定，方向
可能只优化 judge 喜欢的格式。应使用人工抽样、程序化指标、不同 judge 和未参与方向
构造的任务交叉检查，并报告每个切片的分母。

### 12.8 线性方向与非线性行为

高层行为常常是条件化的。一个“更谨慎”方向可能只在模型已经识别出高风险主题后有效，
在普通问题上却造成过度拒答；多个方向叠加还可能出现非线性相互作用。不能把单个方向
在单任务上的收益外推到任意输入，更不能通过简单加权把冲突目标压成一个可靠总分。强度
扫描应包含方向组合、不同层位和足够的正常边界样本。

### 12.9 安全双用和信息暴露

公开控制方法可能帮助研究者理解模型，也可能暴露模型的控制面。章节、博客和 system
card 可以公布抽象方法、风险和指标；内部材料再保存具体向量、hook 和失败 trace；生产
接口则只允许受控 profile。对高风险模型，向量访问权限、审计日志和撤销机制应和权重
访问同等级管理。

### 12.10 不应把表示控制当作权限控制

即使 steering 使模型更谨慎，服务端仍要验证用户、租户、资源范围、工具参数和提交确认。
表示控制解决的是模型倾向，权限控制解决的是系统能否执行。两者的失败模式不同：前者
可能漏拒或误拒，后者可能产生真实副作用；任何一层都不能用另一层的平均分抵消。

## 13. 未来可能演化

### 13.1 从手工方向到自动发现

未来可能自动发现大量行为方向和安全相关 feature。

自动发现可以降低候选搜索成本，但也会放大多重比较和标签泄漏。系统若从数万个方向中
挑出“最能提高某个 benchmark”的一个，必须保留搜索空间、选择标准和独立复验，否则
方向可能只是评估集记忆。自动方法适合提出候选，不应跳过反例和行为回归。

### 13.2 从单方向到多维控制

复杂行为可能不是单一方向，而是多个 feature 和 circuit 共同作用。

多维控制需要处理方向相关性、冲突目标和强度交互。把 honesty、harmlessness、brevity
三个方向分别调大，可能得到更短但更保守的答案，也可能让模型回避必要的解释。未来的
方法需要记录可行区域和副作用曲线，而不是只寻找一个“最佳向量”。

### 13.3 和 SAE 结合

SAE 可以提供更细粒度 feature。

Steering 可以从粗方向转向 feature-level control。

这会带来更精细的控制，也带来更高的验证负担。SAE feature 的标签、字典版本和残差
保真度必须与 steering profile 绑定；feature 在一个层的语义变化，不等于另一个层有
相同含义。解释、监控和干预三种用途仍应分开验证。

### 13.4 和安全监控结合

内部表示监控可能成为未来 safety monitor 的一部分。

内部监控适合做风险升级信号，不适合独立决定高影响动作。它需要和输出策略、工具审计、
人工复核及回滚机制组合；还要测监控信号的漏报、误报、延迟和版本漂移。否则监控本身
会制造一种虚假的安全感。

### 13.5 和系统治理结合

最终 steering 可能作为模型治理工具之一，但仍需要权限、红队、评估和审计配合。

治理层还要规定谁可以创建、批准、启用和撤销 steering profile，哪些模型和业务允许使用，
profile 何时过期，如何响应事故。把向量当作“配置”而不是“代码”会掩盖它对模型行为
的影响；更合理的做法是把它纳入模型 artifact 清单和变更审查。

## 14. 案例：企业知识助手的事实性 steering

一家企业知识助手已经有稳定的 RAG、引用和工具权限层，但在长文档和跨部门问题上仍有
两类错误：一类回答很流畅却把不确定内容说得过于肯定，另一类为了避免幻觉而拒绝了
本来有充分证据的正常问题。团队希望先用一个可开关的 steering profile 做离线实验，
而不是立即重新训练生产模型。

本案例使用 `supported_claim`、`unsupported_claim`、`normal_boundary`、`tool_action` 等
抽象标签，不包含真实企业秘密或高风险操作细节。steering 的目标是提高证据约束下的
回答质量，同时保持正常帮助性和工具授权边界。

### 14.1 任务契约和基线

团队先定义四种结果：回答是否正确，引用是否支持 claim，模型是否表达了必要的不确定性，
以及是否提出了越权工具请求。基线模型在未干预条件下执行所有任务，策略服务和工具权限
保持不变；steering 只作用于离线白盒副本的指定层和 token 位置。

每个任务保留 evidence pack、标准答案或可接受答案范围、引用判定规则和安全处置规则。
不能只用“语气更谨慎”作为目标，因为谨慎措辞可能掩盖引用仍然不支持结论。正常边界
任务尤其重要：它们检验 steering 是否把所有不确定问题都推向拒答。

### 14.2 构造方向和竞争性假设

正例使用有充分证据且正确表达不确定性的回答，负例使用抽象的无证据断言；两组样本在
长度、主题、语言、引用数量和语气上尽量匹配。研究者在多个层位计算激活差，并保留
训练外的方向验证集。

实验同时比较三个假设：

1. 方向确实增强了证据约束表示，因而引用支持率和不确定性表达改善。
2. 方向只学到了固定的谨慎语气，文本看起来更保守，但事实和引用没有改善。
3. 主要问题来自 RAG 证据选择或 judge 判定，激活方向并不是根因。

每个假设都要有反事实预测。比如第二个假设预期正常任务拒答率会上升而引用支持率不
变；第三个假设预期替换检索结果后行为会大幅变化，而 steering 作用较小。

### 14.3 强度扫描和评估

团队冻结一个候选层位，在 \(\alpha=0\)、低、中、高四个强度下执行配对任务。每个强度
记录目标收益、引用支持率、正常帮助性、误拒、工具参数正确率、越权尝试率、延迟和成本。
随机方向、零向量和只改变解码温度的版本作为对照。

如果中等强度提高了引用支持率，低强度几乎没有变化，高强度却让正常边界误拒增加，
那么结论不是“高强度最好”，而是存在一个需要在独立 holdout 上复验的剂量—反应区间。
对多轮 RAG/Agent 任务，还要检查前一轮的 steering 是否污染后续记忆和工具计划。

### 14.4 结果解读

假设离线结果显示：中等强度使 unsupported claim 降低，引用支持率上升，正常帮助性
略有下降，工具越权没有改善，延迟增加。这个结果支持“方向对回答表达和证据约束有局部
影响”，但不支持“模型已经更安全”或“工具风险已解决”。工具权限层仍必须单独保留，
因为 steering 没有改变服务端授权。

如果跨语言和跨领域 holdout 上只有语气变化，引用支持率没有提升，则应撤销该 profile，
回到样本构造、检索和判定器校准。若方向对正常任务和高风险抽象任务都显著提高拒答，
则需要优先处理 over-refusal，而不是继续增加强度。

### 14.5 修复、回退和长期维护

可接受的 profile 应保存向量版本、层位、强度范围、适用任务、已知副作用、评估集版本
和撤销条件。生产启用时先限定到无工具或只读工具场景，使用 shadow 或小范围灰度观察
真实切片；任何越权工具迹象都由服务端拒绝并记录，不依赖 steering 自己纠正。

当模型 checkpoint、tokenizer、量化或 RAG 模板升级时，旧向量自动失效，必须重新做
配对评估。维护成本、hook 失败回退和人工复核也计入单位成功任务成本。这样 steering
才被当作有版本、有边界的模型 artifact，而不是散落在服务代码里的神秘常数。

这个案例的最终结论可以很具体：在限定的知识问答范围内，某个 profile 值得进入 holdout
和只读灰度；工具授权、注入防御和高风险请求处理仍由独立系统控制负责。控制效果与安全
保证被明确分开。

## 15. 资料与证据边界

Steering 研究通常在特定模型、层位、数据集和推理实现上成立。论文可以支持方法和实验
结果，不能自动支持生产模型的安全结论；工具文档可以支持接口和复现路径，不能支持
方向在新 checkpoint 上仍然有效。引用资料时要把方法事实、实验结果和本项目推断分栏。

### 15.1 表示工程与激活控制

- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)：提出从高层行为和表示方向出发的 top-down 方法；结论依赖论文中的模型、任务和表示提取条件。
- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)：研究无需参数更新的激活添加；其强度、层位和行为效果需要在具体模型上重新测量。
- [Contrastive Activation Addition](https://arxiv.org/abs/2312.07569)：讨论从对比激活差构造控制方向的方法；正负样本分布和评估器会直接影响方向含义。
- [Inference-Time Intervention](https://arxiv.org/abs/2306.03341)：研究在推理时干预表示以改善特定行为；它支持局部实验，不替代训练、权限和安全评估。

### 15.2 拒答、事实性与安全边界

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)：研究开源模型拒答行为中的方向性现象；“单一方向”是论文作用域内的实验描述，不是所有模型的安全架构定律。
- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/)：提供激活提取和干预工具；接口、模型支持、hook 语义和版本应按当前文档核对。
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：支持风险治理、测量和管理的组织框架；它不定义某个 steering vector 的效果阈值。

公开资料中的“增强诚实”“降低有害输出”必须补充模型版本、任务分布、样本分母、强度、
解码参数、对照、人工/judge 判定、正常边界和工具权限。没有这些条件，单一 improvement
数字只能作为研究线索，不能写成生产安全事实。

## 16. 常见误区

### 16.1 误区：Steering 就是 prompt engineering

纠正：Prompt 是输入层控制，activation steering 是表示层控制。

### 16.2 误区：找到方向就等于理解机制

纠正：方向可能有效，但不代表完整解释了内部 circuit。

### 16.3 误区：Steering 没有副作用

纠正：它可能改变其他能力、风格、安全边界和泛化表现。

### 16.4 误区：拒答方向说明安全很容易解决

纠正：它说明某些模型拒答行为可能有简洁表示，但也说明安全机制可能脆弱。

### 16.5 误区：Steering 可以替代系统安全

纠正：Steering 是模型内部控制工具，不能替代权限、沙箱、审计和发布条件。

## 17. 小练习

### 练习 1

用自己的话解释 prompt steering、fine-tuning 和 activation steering 的区别。

### 练习 2

设计一个 steering vector 实验，用于增强模型回答中的事实性。

要求说明正负样本、激活层选择、强度系数、评估指标和副作用检查。

### 练习 3

解释为什么拒答方向这类发现同时有安全价值和双用风险。

### 练习 4

比较 representation engineering 和 mechanistic interpretability。

要求说明 top-down 与 bottom-up 的区别。

### 练习 5

为一个企业助手设计 steering 上线评估。

要求覆盖：helpfulness、factuality、refusal、over-refusal、latency、jailbreak 和人工审核。

## 18. 最小可运行 Steering 审计 demo

下面的 demo 不需要真实模型，只用 toy 激活和 toy 评估表演示 steering 的核心审计流程。真实项目中，`positive_activations`、`negative_activations` 和 `case["activation"]` 应来自模型 forward hook。

```python
from math import sqrt


def mean_vector(rows):
    if not rows:
        return None
    dims = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(dims)]


def subtract(a, b):
    if a is None or b is None:
        return None
    return [x - y for x, y in zip(a, b)]


def add(a, b):
    if a is None or b is None:
        return None
    return [x + y for x, y in zip(a, b)]


def scale(alpha, v):
    if v is None:
        return None
    return [alpha * x for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return sqrt(dot(v, v))


def normalize(v, eps=1e-12):
    if v is None:
        return None
    length = norm(v)
    if length <= eps:
        return None
    return [x / length for x in v]


def projection_score(h, v):
    if h is None or v is None:
        return None
    length = norm(h)
    if length <= 1e-12:
        return None
    return dot(h, v) / length


def round_vector(v):
    if v is None:
        return None
    return [round(x, 3) for x in v]


positive_activations = [
    [1.2, 0.8, -0.1],
    [1.0, 0.7, 0.0],
    [1.1, 0.9, 0.1],
]
negative_activations = [
    [0.1, -0.2, 0.8],
    [0.0, -0.1, 0.7],
    [0.2, -0.3, 0.9],
]

mu_pos = mean_vector(positive_activations)
mu_neg = mean_vector(negative_activations)
steering_vector = subtract(mu_pos, mu_neg)
unit_vector = normalize(steering_vector)

cases = [
    {
        "id": "factual_qa",
        "activation": [0.4, 0.2, 0.4],
        "target_gain": 0.30,
        "safe_gain": 0.00,
        "side_loss": 0.03,
        "over_refusal": 0.00,
    },
    {
        "id": "citation_answer",
        "activation": [0.5, 0.3, 0.3],
        "target_gain": 0.24,
        "safe_gain": 0.00,
        "side_loss": 0.02,
        "over_refusal": 0.00,
    },
    {
        "id": "boundary_safety",
        "activation": [0.2, 0.0, 0.7],
        "target_gain": 0.20,
        "safe_gain": 0.38,
        "side_loss": 0.05,
        "over_refusal": 0.05,
    },
    {
        "id": "normal_help",
        "activation": [0.3, 0.1, 0.2],
        "target_gain": 0.05,
        "safe_gain": 0.00,
        "side_loss": 0.08,
        "over_refusal": 0.08,
    },
    {
        "id": "tool_caution",
        "activation": [0.1, 0.2, 0.5],
        "target_gain": 0.18,
        "safe_gain": 0.22,
        "side_loss": 0.04,
        "over_refusal": 0.03,
    },
]


def evaluate_alpha(alpha):
    if unit_vector is None:
        return {
            "alpha": alpha,
            "avg_projection_shift": None,
            "target_uplift": None,
            "safe_gain": None,
            "side_drop": None,
            "over_refusal_delta": None,
            "objective": None,
        }

    projection_shift = []
    target_uplift = []
    safe_gain = []
    side_drop = []
    over_delta = []

    for case in cases:
        before = projection_score(case["activation"], unit_vector)
        steered = add(case["activation"], scale(alpha, unit_vector))
        after = projection_score(steered, unit_vector)
        projection_shift.append(after - before)
        target_uplift.append(alpha * case["target_gain"])
        safe_gain.append(alpha * case["safe_gain"])
        side_drop.append(alpha * case["side_loss"])
        over_delta.append(alpha * case["over_refusal"])

    def mean_or_none(values):
        if not values or any(value is None for value in values):
            return None
        return round(sum(values) / len(values), 3)

    metrics = {
        "alpha": alpha,
        "avg_projection_shift": mean_or_none(projection_shift),
        "target_uplift": mean_or_none(target_uplift),
        "safe_gain": mean_or_none(safe_gain),
        "side_drop": mean_or_none(side_drop),
        "over_refusal_delta": mean_or_none(over_delta),
    }
    values = [
        metrics["target_uplift"],
        metrics["safe_gain"],
        metrics["side_drop"],
        metrics["over_refusal_delta"],
    ]
    metrics["objective"] = (
        round(
            metrics["target_uplift"]
            + metrics["safe_gain"]
            - 0.8 * metrics["side_drop"]
            - 1.2 * metrics["over_refusal_delta"],
            3,
        )
        if all(value is not None for value in values)
        else None
    )
    return metrics


alpha_grid = [0.0, 0.4, 0.8, 1.2]
scan = [evaluate_alpha(alpha) for alpha in alpha_grid]
valid_scan = [item for item in scan if item["objective"] is not None]
best = max(valid_scan, key=lambda item: item["objective"]) if valid_scan else None

thresholds = {
    "target": {"operator": ">=", "value": 0.10},
    "safe": {"operator": ">=", "value": 0.08},
    "side_effect": {"operator": "<=", "value": 0.06},
    "over_refusal": {"operator": "<=", "value": 0.05},
    "projection": {"operator": ">=", "value": 0.20},
}

signals = {
    "target": best["target_uplift"] if best else None,
    "safe": best["safe_gain"] if best else None,
    "side_effect": best["side_drop"] if best else None,
    "over_refusal": best["over_refusal_delta"] if best else None,
    "projection": best["avg_projection_shift"] if best else None,
}


def meets_threshold(signal, threshold):
    if signal is None:
        return False
    if threshold["operator"] == ">=":
        return signal >= threshold["value"]
    if threshold["operator"] == "<=":
        return signal <= threshold["value"]
    raise ValueError(f"unsupported operator: {threshold['operator']}")


evidence_status = {
    name: meets_threshold(signals[name], threshold)
    for name, threshold in thresholds.items()
}

undefined_metrics = [name for name, value in signals.items() if value is None]

actions = {
    "target": "replicate_target_gain_on_holdout",
    "safe": "keep_high_risk_scope_restricted_until_red_team_retest",
    "side_effect": "reduce_alpha_or_rebuild_direction",
    "over_refusal": "expand_normal_boundary_and_safe_completion_eval",
    "projection": "inspect_layer_and_position_specificity",
}

decision = {
    "scope": "offline_profile_only",
    "status": "hold_for_holdout_and_read_only_shadow",
    "evidence_status": evidence_status,
    "undefined_metrics": undefined_metrics,
    "next_actions": list(actions.values()),
}

print("mu_pos=", round_vector(mu_pos))
print("mu_neg=", round_vector(mu_neg))
print("steering_vector=", round_vector(steering_vector))
print("unit_vector=", round_vector(unit_vector))
print("scan=", scan)
print("best=", best)
print("thresholds=", thresholds)
print("signals=", signals)
print("evidence_status=", evidence_status)
print("undefined_metrics=", undefined_metrics)
print("actions=", actions)
print("decision=", decision)
```

预期输出：

```text
mu_pos= [1.1, 0.8, 0.0]
mu_neg= [0.1, -0.2, 0.8]
steering_vector= [1.0, 1.0, -0.8]
unit_vector= [0.615, 0.615, -0.492]
scan= [{'alpha': 0.0, 'avg_projection_shift': 0.0, 'target_uplift': 0.0, 'safe_gain': 0.0, 'side_drop': 0.0, 'over_refusal_delta': 0.0, 'objective': 0.0}, {'alpha': 0.4, 'avg_projection_shift': 0.468, 'target_uplift': 0.078, 'safe_gain': 0.048, 'side_drop': 0.018, 'over_refusal_delta': 0.013, 'objective': 0.096}, {'alpha': 0.8, 'avg_projection_shift': 0.671, 'target_uplift': 0.155, 'safe_gain': 0.096, 'side_drop': 0.035, 'over_refusal_delta': 0.026, 'objective': 0.192}, {'alpha': 1.2, 'avg_projection_shift': 0.752, 'target_uplift': 0.233, 'safe_gain': 0.144, 'side_drop': 0.053, 'over_refusal_delta': 0.038, 'objective': 0.289}]
best= {'alpha': 1.2, 'avg_projection_shift': 0.752, 'target_uplift': 0.233, 'safe_gain': 0.144, 'side_drop': 0.053, 'over_refusal_delta': 0.038, 'objective': 0.289}
thresholds= {'target': {'operator': '>=', 'value': 0.1}, 'safe': {'operator': '>=', 'value': 0.08}, 'side_effect': {'operator': '<=', 'value': 0.06}, 'over_refusal': {'operator': '<=', 'value': 0.05}, 'projection': {'operator': '>=', 'value': 0.2}}
signals= {'target': 0.233, 'safe': 0.144, 'side_effect': 0.053, 'over_refusal': 0.038, 'projection': 0.752}
evidence_status= {'target': True, 'safe': True, 'side_effect': True, 'over_refusal': True, 'projection': True}
undefined_metrics= []
actions= {'target': 'replicate_target_gain_on_holdout', 'safe': 'keep_high_risk_scope_restricted_until_red_team_retest', 'side_effect': 'reduce_alpha_or_rebuild_direction', 'over_refusal': 'expand_normal_boundary_and_safe_completion_eval', 'projection': 'inspect_layer_and_position_specificity'}
decision= {'scope': 'offline_profile_only', 'status': 'hold_for_holdout_and_read_only_shadow', 'evidence_status': {'target': True, 'safe': True, 'side_effect': True, 'over_refusal': True, 'projection': True}, 'undefined_metrics': [], 'next_actions': ['replicate_target_gain_on_holdout', 'keep_high_risk_scope_restricted_until_red_team_retest', 'reduce_alpha_or_rebuild_direction', 'expand_normal_boundary_and_safe_completion_eval', 'inspect_layer_and_position_specificity']}
```

这个 demo 对应真实项目中的关键点：

1. Steering vector 来自正负行为样本激活差，而不是手写规则。
2. 强度 \(\alpha\) 要扫描，不能凭直觉选。
3. 目标收益、安全收益、副作用和误拒增量要同时看。
4. `decision` 只把当前结果限定为离线 profile 和只读 shadow 的下一步，不代表真实模型可以直接上线。

## 19. 本章总结

Steering 是控制模型行为的一类方法，从 prompt 控制、解码控制，到激活层和表示层控制都有不同形式。

Representation Engineering 从高层行为出发，寻找模型内部可监控、可干预的表示方向，是一种 top-down transparency 思路。

Steering vector 通常通过正负样本激活差异得到，推理时加入模型激活以调节行为。

Contrastive Activation Addition 是 activation steering 的代表方法之一。

拒答方向说明模型安全行为可能有可分析的内部表示，但也暴露了安全机制的潜在脆弱性和双用风险。

Steering 的价值在于快速、可开关、适合研究和辅助控制；局限在于泛化、副作用、部署复杂度和安全双用风险。

真实系统中，steering 应和对齐训练、安全评估、红队、权限控制、系统监控和治理流程结合，而不是单独承担安全保证。
