# 第六章：Diffusion 基础——从随机路径到可控采样

Diffusion 是一类把生成问题改写成“沿着一条随机路径逐步还原数据”的模型。训练时，我们从真实样本出发，按照事先规定的噪声过程得到不同噪声强度的样本，再训练网络预测噪声、干净样本、速度或 score。生成时没有真实样本可供参考，于是从随机噪声出发，反复调用这个网络和采样器，沿反方向回到数据分布附近。

这句话仍然只是入口。真正理解 diffusion，至少要分清：

1. forward process 是人为规定的加噪分布；
2. denoiser 是需要学习的函数；
3. scheduler 或 solver 规定如何使用 denoiser 走反向路径；
4. condition 决定路径朝哪个目标区域移动；
5. U-Net、DiT 等是实现 denoiser 的网络结构；
6. pixel diffusion 和 latent diffusion 只是状态空间不同；
7. DDPM、DDIM、SDE、consistency 和 flow matching 并不是同一个公式换了名字。

本章从 DDPM 的离散高斯链开始，逐步连接到 score、采样器、CFG、latent diffusion、DiT 和相邻的新路线。每个公式都说明变量、适用范围和工程后果；最后用一个只依赖 Python 标准库的向量实验，检查闭式加噪、噪声预测、x0 重建、反向均值、score、CFG 和 latent 成本的算术是否一致。

## 0. 研究对象、资料与边界

### 0.1 本章回答什么问题

本章要回答的不是“哪一个生成模型最好”，而是以下几个基础问题：

- 为什么可以把真实图片逐步加到近似高斯噪声；
- 为什么训练时可以随机抽一个时间步，而不必真的走完整条加噪链；
- 预测噪声、预测 x0、预测 velocity 和预测 score 如何互相转换；
- 反向高斯均值从哪里来，采样时为什么还要加随机项；
- scheduler 为什么既影响质量又影响速度；
- 文本条件如何进入 U-Net 或 DiT；
- CFG 为什么会增强 prompt 遵循，也为什么会带来过饱和和多样性损失；
- latent diffusion 为什么降低空间成本，却不能消除 VAE 和 decoder 的代价；
- DiT、蒸馏、consistency 和 flow matching 与 DDPM 的关系在哪里。

### 0.2 资料层级

| 资料 | 适合支持的判断 | 不应直接推出的判断 |
| --- | --- | --- |
| DDPM 原始论文 | 离散加噪链、变分目标和噪声预测路线 | 任意 scheduler 都等价 |
| DDIM 原始论文 | 非马尔可夫采样和确定性路径 | 少步一定保持同样质量 |
| Score SDE 论文 | 连续时间 score/SDE 视角 | 某个离散实现自动满足连续极限 |
| Classifier Guidance 与 CFG 论文 | 条件引导的数学构造 | guidance scale 在所有模型上同样可用 |
| Latent Diffusion 论文 | 在压缩表示空间扩散的路线 | 任意 VAE 压缩都不损伤语义 |
| DiT 论文 | Transformer 作为 denoiser 的结构实验 | U-Net 在所有尺度都应被替换 |
| Diffusers scheduler 文档 | 某个版本的 scheduler API 与参数语义 | API 默认值就是科学结论 |
| 本章 demo | 小向量上的公式和算术 | 真实图片质量、审美、文本遵循或安全 |

论文结论必须绑定论文中的模型、数据、训练预算和评估协议。官方文档可以说明接口如何工作，却不能替目标设备上的吞吐、峰值显存和图像质量测量。

### 0.3 小白先建立一张地图

可以先把 diffusion 想成四个角色：

- 数据点 x0：希望最后生成的清晰样本；
- 噪声 xt：在某个时间步看到的中间状态；
- denoiser：根据 xt、时间步和条件，估计下一步该往哪里走；
- scheduler：决定每一步走多大、是否保留随机性。

专家还要补充五个维度：

- 训练状态空间是像素还是 latent；
- 网络预测的是 epsilon、x0、v 还是 score；
- 训练时间采样和推理时间网格是否匹配；
- sampler 的数值阶数、随机性和 solver 假设；
- 条件引导是否改变质量、多样性、对齐和成本。

### 0.4 先说一个容易误导的比喻

“加噪再去噪”是有用的直觉，但它不意味着模型在生成时真的知道原始图片。训练时使用 x0 构造监督；推理时只拥有当前 xt、时间步和条件。模型学到的是数据分布下的统计方向，不是某一张训练图片的逆向录像。

同样，“预测噪声”也不是说输出一定就是人眼看到的随机颗粒。epsilon 是公式中用于构造 xt 的高维随机变量；网络输出的是对这个变量的估计，估计误差会影响 x0 重建和后续采样。

## 1. 生成模型与扩散路径

### 1.1 生成的目标是分布而不是复制

设数据样本为 x，真实数据分布为 p_data(x)。生成模型希望构造一个可采样的分布 p_theta(x)，使它在结构、语义和统计特征上接近 p_data。

无条件生成可以写成：

~~~math
x\sim p_\theta(x).
~~~

文本到图像生成则是条件分布：

~~~math
x\sim p_\theta(x\mid c),
~~~

其中 c 可以是文本 embedding、类别、边缘图、深度图、姿态或其他条件。

“接近真实分布”不等于逐像素复制训练样本。一个生成系统可能在某些切片上表现好，在小字、手指、计数、版权相似性或长尾概念上表现差。生成质量必须按目标任务测量。

### 1.2 与其他生成路线的区别

自回归图像模型把图片编码成离散 token，然后按顺序预测：

~~~text
z_1 -> z_2 -> z_3 -> ... -> z_n
~~~

GAN 让生成器直接产生样本，并通过判别器提供对抗反馈。VAE 通过潜变量和重构/分布正则学习可采样表示。Diffusion 则在连续或近似连续的状态上定义一条加噪路径，再学习逆向转换。

这些是建模和优化路径的区别，不是“一个生成清晰图片，另一个不能”。不同路线可能结合使用，例如在 latent 空间中使用 diffusion，或用 Transformer 作为 diffusion denoiser。

### 1.3 为什么扩散训练容易构造监督

给定真实样本 x0，我们可以自己采样一个高斯噪声 epsilon，并按已知公式得到 xt。于是训练目标中的“正确答案”是我们刚刚采样的 epsilon，而不是依赖人工为每张图片写一个去噪标签。

这种监督有三个性质：

1. 加噪过程可计算；
2. 任意时间步都能构造训练对；
3. 网络输出可以用均方误差或其他明确目标比较。

但监督可计算不等于问题简单。网络仍要同时处理空间结构、时间步、条件、分辨率、长尾概念和采样误差。

## 2. DDPM 的 forward diffusion

### 2.1 一步加噪的定义

令 x0 为干净样本，时间步 t 从 1 到 T。DDPM 使用一组噪声方差 beta_t，通常满足：

~~~math
0<\beta_t<1.
~~~

定义保留信号的比例：

~~~math
\alpha_t=1-\beta_t.
~~~

一步 forward transition 是：

~~~math
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{\alpha_t}\,x_{t-1},
\beta_t I
\right).
~~~

等价的采样写法是：

~~~math
x_t
=
\sqrt{\alpha_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I).
~~~

这里 I 是与样本维度匹配的单位矩阵。alpha_t 越接近 1，当前一步保留的信号越多；beta_t 越大，当前一步加入的噪声越多。

本节默认时间步是整数 $t\in\{1,\ldots,T\}$，并要求每个 $\beta_t$ 都是有限实数且满足 $0<\beta_t<1$。如果 $\beta_t=0$，这一步没有注入噪声；如果 $\beta_t\geq1$，$\alpha_t$ 不再是合法的正信号系数。实际系统还要检查 schedule 生成的 $\bar{\alpha}_t$ 是否保持在 $(0,1)$，因为后面的 $s_t=\sqrt{1-\bar{\alpha}_t}$ 和 $\hat{x}_0$ 都依赖这些定义域。

### 2.2 累积信号比例

定义累计乘积：

~~~math
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s.
~~~

它表示从第 1 步到第 t 步累积保留下来的信号比例。随着 t 增大，通常有 bar alpha_t 逐步下降。

将每一步的高斯变换合并，可以得到闭式边缘分布：

~~~math
q(x_t\mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}\,x_0,
(1-\bar{\alpha}_t)I
\right).
~~~

因此可以直接采样：

~~~math
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
~~~

这个闭式公式是训练效率的关键。训练时随机抽取 t，只需一次前向加噪，就能构造 xt，而不必从 x1 一直模拟到 xt。

### 2.3 信号、噪声与 SNR

令：

~~~math
a_t=\sqrt{\bar{\alpha}_t},
\qquad
s_t=\sqrt{1-\bar{\alpha}_t}.
~~~

则：

~~~math
x_t=a_t x_0+s_t\epsilon.
~~~

在一个简化的单位方差假设下，信号与噪声的强度比可以用：

~~~math
\mathrm{SNR}(t)
=
\frac{\bar{\alpha}_t}
{1-\bar{\alpha}_t}
=
\frac{a_t^2}{s_t^2}
~~~

表示。早期时间步 SNR 较高，xt 仍保留较多结构；晚期时间步 SNR 较低，样本更接近噪声。

SNR 不是图像质量分数。它只是描述给定 forward schedule 下的信号噪声比例，不能单独说明模型在该时间步预测得好不好。

### 2.4 时间表的选择

beta schedule 决定了训练样本在不同时间步的难度。常见设计包括线性变化、余弦变化和连续噪声水平参数化。

选择 schedule 时需要同时考虑：

- 早期时间步是否覆盖足够清晰的样本；
- 晚期时间步是否真的接近目标噪声分布；
- 训练时的时间采样概率；
- 推理时选取的离散时间网格；
- 目标分辨率和状态空间的统计性质。

不同 schedule 改变的不只是一个数组。它会改变 xt 的分布、网络在各时间步看到的任务难度、loss 的权重和 sampler 的假设。不能把一个 schedule 的 beta 数值直接移植到另一个归一化范围或 latent 分布上。

### 2.5 一个标量直觉例子

假设某个时间步 bar alpha_t=0.81，则：

~~~math
a_t=\sqrt{0.81}=0.9,
\qquad
s_t=\sqrt{0.19}\approx0.4359.
~~~

如果某个标量 x0=2，抽到的噪声 epsilon=0.5，则：

~~~math
x_t
=
0.9\times2+0.4359\times0.5
\approx2.01795.
~~~

这里 xt 仍然靠近 x0，是因为该时间步的信号系数较大。若 bar alpha_t 降到 0.01，则信号系数只有 0.1，噪声项的系数约为 0.995，xt 的随机性会明显增强。

## 3. 训练目标与参数化

### 3.1 epsilon prediction

最常见的入门目标是让网络预测 forward 时采样的 epsilon：

~~~math
\mathcal{L}_{\epsilon}
=
\mathbb{E}_{x_0,t,\epsilon,c}
\left[
\left\|
\epsilon
-
\epsilon_\theta(x_t,t,c)
\right\|_2^2
\right].
~~~

其中：

- x0 是真实样本；
- t 是随机时间步；
- epsilon 是构造 xt 的高斯噪声；
- c 是条件，可以为空；
- epsilon_theta 是 denoiser 输出。

MSE 的优势是目标清晰、实现简单。它并不意味着最终评估只看 MSE；较低的噪声误差不一定对应更好的文本遵循、构图、细节或审美。

### 3.2 从 epsilon 预测恢复 x0

由 xt=a_t x0+s_t epsilon，可以解出：

~~~math
\hat{x}_0
=
\frac{x_t-s_t\epsilon_\theta(x_t,t,c)}
{a_t}.
~~~

展开原符号就是：

~~~math
\hat{x}_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\,
\epsilon_\theta(x_t,t,c)
}{
\sqrt{\bar{\alpha}_t}
}.
~~~

当 bar alpha_t 很小时，分母 a_t 很小，噪声预测中的微小误差会被放大到 x0_hat。因此晚期时间步的 x0 重建可能数值敏感，实际系统会使用合适的参数化、裁剪或 scheduler 处理。

这个换算还要求 $a_t>0$，也就是 $\bar{\alpha}_t>0$；如果实现把最后一个时间步直接设成 $\bar{\alpha}_t=0$，就不能继续使用这个除法，而应采用该 scheduler 为端点定义的特殊处理。类似地，$x_0$ prediction 到 $\epsilon$ prediction 的换算要求 $s_t>0$，所以不能把无噪声的 $t=0$ 状态和带噪时间步混用。

### 3.3 x0 prediction

另一种参数化直接让网络预测 x0：

~~~math
\hat{x}_0=f_\theta(x_t,t,c).
~~~

再由：

~~~math
\hat{\epsilon}
=
\frac{x_t-a_t\hat{x}_0}{s_t}
~~~

得到噪声估计。x0 prediction 对某些任务和损失权重有直观性，但当 s_t 很小时，epsilon 的换算会敏感。选择参数化时要考虑时间步范围、目标损失、数值精度和采样器要求。

### 3.4 v prediction

定义：

~~~math
v_t
=
a_t\epsilon-s_t x_0.
~~~

由于 xt=a_t x0+s_t epsilon，可以反解出：

~~~math
x_0=a_t x_t-s_t v_t,
~~~

~~~math
\epsilon=s_t x_t+a_t v_t.
~~~

因此 v prediction 不是凭空增加一个目标，而是选择了另一组旋转后的坐标。它可能让不同信噪比区域的训练和采样更容易平衡，但实际效果仍取决于 schedule、loss weighting、数据和实现。

### 3.5 score prediction

对条件 forward 分布 q(x_t|x0)，其 score 是：

~~~math
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-a_t x_0}{s_t^2}
=
-\frac{\epsilon}{s_t}.
~~~

因此在这个条件高斯关系下，epsilon prediction 可以转换成 score 估计：

~~~math
s_\theta(x_t,t,c)
\approx
-\frac{\epsilon_\theta(x_t,t,c)}{s_t}.
~~~

需要注意，这个公式写的是给定 x0 的条件高斯 score。连续 score-based 生成模型讨论的往往是边缘分布 p_t(x) 的 score，二者通过训练分布和条件期望联系起来，但不能不加说明地当成完全相同的对象。

### 3.6 loss weighting 与时间采样

最简单的 MSE 对所有采样到的时间步使用相同形式，但不同时间步的信号噪声比例和任务难度不同。工程上可能使用：

- 改变时间步采样概率；
- 对不同时间步加权；
- 根据 SNR 调整损失；
- 选择 epsilon、x0 或 v 参数化；
- 分别监控高 SNR 和低 SNR 切片。

任何 weighting 都改变了训练目标。若只报告一个总体 loss，无法判断模型是否在某些时间区域退化。至少应记录时间步直方图、分桶 loss、重建误差和采样质量。

### 3.7 随机时间步训练伪代码

~~~text
for each clean sample x0 and condition c:
    sample timestep t
    sample epsilon from N(0, I)
    compute xt = a_t * x0 + s_t * epsilon
    predict epsilon_hat = denoiser(xt, t, c)
    optimize distance(epsilon_hat, epsilon)
~~~

这段流程中没有真实的 reverse sample。训练使用的是已知的 forward 构造和目标噪声；reverse 采样在推理阶段才逐步展开。

## 4. Reverse process：从噪声回到数据

### 4.1 为什么需要学习反向分布

forward transition q(x_t|x_{t-1}) 是我们定义的，但直接反过来写 q(x_{t-1}|x_t) 还需要知道真实数据分布。模型学习的是一个近似反向分布：

~~~math
p_\theta(x_{t-1}\mid x_t,c)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t,c),
\Sigma_\theta(x_t,t,c)
\right).
~~~

最简单的 DDPM 讲解通常先固定或预先规定方差，让网络主要预测均值所需的信息。

### 4.2 已知 x0 时的后验

在 DDPM 的高斯链中，给定 x0 和 xt，真实后验仍然是高斯：

~~~math
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right).
~~~

其中：

~~~math
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}
{1-\bar{\alpha}_t}
\beta_t,
~~~

~~~math
\tilde{\mu}_t(x_t,x_0)
=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}
{1-\bar{\alpha}_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}
{1-\bar{\alpha}_t}x_t.
~~~

推理时没有真实 x0，所以用网络的 epsilon、x0 或 v 估计替代它，构造 mu_theta。

### 4.3 epsilon 参数化下的反向均值

使用 epsilon prediction 时，常见的均值写法是：

~~~math
\mu_\theta(x_t,t,c)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t,c)
\right).
~~~

这一公式中的 beta_t、alpha_t 和 bar alpha_t 必须来自同一套 schedule。若训练和推理使用不同的索引、归一化或时间网格，公式虽然能运行，数值路径却可能已经不再匹配。

### 4.4 反向采样的随机项

反向一步可以写成：

~~~math
x_{t-1}
=
\mu_\theta(x_t,t,c)
+
\sigma_t z,
\qquad
z\sim\mathcal{N}(0,I).
~~~

当 t>1 时通常保留随机项；到最后一步可以令 z=0，避免把已经得到的结果再次注入噪声。随机项使同一个条件和不同初始噪声能够产生不同样本，也使采样轨迹不完全由均值决定。

若使用 DDIM 或 ODE 风格采样，更新可以是确定性的或具有不同的随机性。采样器改变的是如何利用 denoiser，并不等于重新训练了 denoiser。

### 4.5 Ancestral sampling 的文字版

~~~text
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    predict epsilon or another parameterization
    construct x0_hat or reverse mean
    sample x_{t-1} with scheduler variance
return x_0
~~~

这里的 x_T 只有在 schedule 和数据归一化匹配时才近似标准高斯。把任意范围的像素或 latent 直接当作与 N(0,I) 匹配，可能导致采样初始分布和训练分布不一致。

## 5. 连续 score 与 SDE 视角

### 5.1 为什么还需要连续时间视角

离散 DDPM 用有限个时间步描述路径，直观且容易实现。连续时间视角把时间记为 t∈[0,1]，用随机微分方程描述：

~~~math
dx=f(x,t)\,dt+g(t)\,dW_t,
~~~

其中 f 是 drift，g 是 diffusion coefficient，W_t 是 Brownian motion。

这种视角可以统一不同噪声路径和 score-based 采样器，但它引入了连续时间、数值积分和边界约定。不能因为写成 SDE，就自动得到更好的图片。

### 5.2 Score 是什么

对时间 t 的边缘分布 p_t(x)，score 定义为：

~~~math
s^\star(x,t)
=
\nabla_x\log p_t(x).
~~~

它指向 log density 增长最快的方向。直观上，在当前噪声水平下，score 告诉我们向哪里移动更可能回到数据高密度区域。

神经网络学习的是：

~~~math
s_\theta(x,t,c)
\approx
\nabla_x\log p_t(x\mid c).
~~~

epsilon prediction 可以通过时间相关的尺度转换为 score prediction，但必须使用与训练 schedule 对应的系数。

### 5.3 Reverse-time SDE

对 forward SDE：

~~~math
dx=f(x,t)\,dt+g(t)\,dW_t,
~~~

reverse-time SDE 的形式可写成：

~~~math
dx=
\left[
f(x,t)-g(t)^2s_\theta(x,t)
\right]dt
+
g(t)d\bar W_t,
~~~

其中反向时间的积分约定很重要。不同论文和代码可能把时间从 1 积到 0，或把负的 dt 隐藏在 solver 中。读公式时必须同时看时间方向和噪声符号。

因此，看到同一个 `reverse_sde` 名称时，不能只比较 drift 的字符串形式。要同时记录时间变量的方向、$dt$ 的符号、score 的定义是条件还是边缘分布，以及 solver 是否已经在内部吸收了反向积分的负号。

### 5.4 Probability flow ODE

同一个 forward diffusion 还可以对应一个确定性的 probability flow ODE：

~~~math
dx=
\left[
f(x,t)-\frac{1}{2}g(t)^2s_\theta(x,t)
\right]dt.
~~~

它没有随机扩散项，但在理想 score 下可以与 SDE 具有相同的边缘分布。实际采样仍会受到 score 误差、数值积分误差、离散步数和初始分布的影响。

SDE、ODE、DDPM 和 DDIM 可以放进同一张概念图中，但实现细节不能互换。尤其是 solver 所需的模型输出参数化、sigma 定义和时间坐标必须匹配。

## 6. Scheduler 与采样器

### 6.1 Scheduler 不是一个 beta 数组

一个 scheduler 可能同时负责：

1. 定义训练或推理使用的时间索引；
2. 保存 alpha、sigma、bar alpha 等系数；
3. 把模型输出转换成 x0、epsilon 或 velocity；
4. 根据当前和下一时间步更新样本；
5. 决定随机项、方差或 solver 阶数；
6. 处理 clipping、scaling 和边界时间步。

因此“换 scheduler”可能改变多件事。实验记录中至少要保存 scheduler 名称、版本、时间步、prediction type、采样步数、guidance scale 和初始噪声种子。

一个可运行的 scheduler 还应对输入做契约检查：时间步不能越过训练 schedule 的范围，prediction type 必须与 denoiser 输出一致，且所有需要开方或作分母的系数必须处在合法区间。把非法系数钳制成一个可运行的数值，可能掩盖训练和推理 schedule 已经不匹配的问题。

### 6.2 DDPM ancestral sampling

DDPM 反向采样通常保留与后验方差相关的随机项。它的优点是概念直接，缺点是往往需要较多步数才能得到稳定质量。

每一步的成本包括一次或多次 denoiser 调用、条件 cross-attention、scheduler 算术和可能的 VAE 操作。若使用 CFG，条件和无条件预测可能需要两次前向，除非实现把它们拼成一个 batch。

### 6.3 DDIM

DDIM 构造了与相同训练目标兼容的另一类采样路径，可以使用更少的离散时间步，并在某些设置下使用确定性采样。它不是“DDPM 的加速开关”这么简单：

- 采样路径的随机性不同；
- 选取的时间步子集不同；
- eta 等参数会影响随机性；
- 少步时数值误差更明显；
- 相同 denoiser 也可能产生不同细节和多样性。

### 6.4 Euler、Heun 与高阶 solver

Euler 方法使用局部一阶近似，简单但步数少时误差较大；Heun 使用额外信息改善局部近似；DPM-Solver 等方法利用 diffusion ODE 的结构，试图在较少函数评估下保持质量。

“高阶”不等于在任何模型上都更好。solver 假设的时间坐标、模型输出类型、噪声参数化和网络误差都会影响最终结果。一个实现若把 v prediction 当成 epsilon 送入只接受 epsilon 的 scheduler，理论阶数再高也没有意义。

### 6.5 采样步数的实际账本

若每一步需要 F 次 denoiser forward，采样步数为 K，单张图的主要 denoiser 调用量约为：

~~~math
N_{\mathrm{forward}}
\approx K F.
~~~

使用 CFG 且未做 batch 合并时，F 可能接近 2；使用蒸馏模型或特殊实现时，F 的定义可能不同。端到端延迟还要加上文本编码、VAE 解码、数据搬运和后处理。

比较两个采样器时，应固定：

- 初始 latent；
- prompt 和 text encoder；
- VAE；
- 分辨率；
- guidance；
- batch；
- 硬件和精度；
- 输出质量指标。

否则“更快”可能只是降低了分辨率或换了初始噪声。

## 7. Denoiser 架构：U-Net 与 DiT

### 7.1 U-Net 的多尺度路径

U-Net 通常包含下采样路径、中间块和上采样路径：

~~~text
noisy image or latent
    -> down blocks
    -> middle blocks
    -> up blocks
    -> predicted epsilon / v / x0
~~~

下采样扩大感受野，帮助模型理解整体结构；上采样恢复空间细节；skip connection 把浅层局部信息传回高分辨率阶段；time embedding 告诉每个 block 当前噪声水平；cross-attention 或其他条件模块注入文本和控制信号。

U-Net 适合多尺度图像结构，但每层的通道、分辨率、注意力位置和条件注入方式都会影响显存和速度。不能只画一个 U 形图就认为结构已经确定。

### 7.2 DiT 的 token 化路径

Diffusion Transformer 通常先把图片或 latent 切成 patch tokens，再用 Transformer block 处理：

~~~math
N
=
\frac{H}{P}\times\frac{W}{P},
~~~

其中 H、W 是 latent 或图片空间的高宽，P 是 patch size。若 H 或 W 不能被 P 整除，还要说明 padding 或裁剪策略。

一个简化的注意力成本项是：

~~~math
C_{\mathrm{attn}}
\propto
L N^2 d,
~~~

其中 L 是 block 数，N 是 token 数，d 是 hidden size。分辨率提高会让 N 增加，注意力中的平方项可能成为主要代价。

DiT 的研究价值在于把 denoiser 置于 Transformer scaling 的框架中，但它不是把 U-Net 的每个细节机械替换成 self-attention。patch size、位置表示、时间条件、调制方式、latent 空间和训练规模共同决定结果。

### 7.3 时间和条件如何进入网络

时间 embedding 可以通过加法、FiLM、AdaLN 或其他调制方式进入 block。文本条件可以通过 cross-attention、拼接 token 或调制信号注入。不同注入点会改变网络在早期粗结构和后期细节阶段使用条件的方式。

因此阅读一个 denoiser 实现时，至少要回答：

- 时间 embedding 在哪些 block 使用；
- 文本条件的 key/value 来自哪里；
- 图像 latent 的空间顺序如何保留；
- 输出通道对应 epsilon、v 还是其他目标；
- scheduler 是否知道这个 prediction type。

### 7.4 图像分辨率与计算

在像素空间，H×W 直接决定状态元素数量；在 latent 空间，VAE 下采样会减少空间尺寸，但通道数、denoiser 宽度和 attention 仍然有成本。对 DiT，patch size 又会改变 N；对 U-Net，分辨率和每层 feature map 会改变卷积和激活显存。

“latent 更小”只能作为第一层估算。完整账本还要包含：

- VAE encoder 和 decoder；
- latent channel；
- denoiser 参数；
- attention 中间量；
- activation；
- CFG 额外 forward；
- 采样步数。

## 8. 条件生成与 cross-attention

### 8.1 文本条件进入 denoiser

文生图的一条常见路径是：

~~~text
prompt
    -> tokenizer
    -> text encoder
    -> text embeddings
    -> denoiser conditioning
    -> generated latent
    -> VAE decoder
~~~

在 cross-attention 中，图像 latent hidden state 作为 query，文本 token 表示作为 key 和 value：

~~~math
Q=H_{\mathrm{latent}}W_Q,
\qquad
K=H_{\mathrm{text}}W_K,
\qquad
V=H_{\mathrm{text}}W_V.
~~~

注意力输出为：

~~~math
\operatorname{Attn}(H_{\mathrm{latent}},H_{\mathrm{text}})
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V.
~~~

这只是说明信息通路，不表示某个词会稳定对应到一个清晰的像素区域。文本 token、空间特征和多层 denoising 共同决定结果。

### 8.2 Classifier guidance 与 classifier-free guidance

Classifier guidance 使用额外分类器的梯度把样本推向目标类别。Classifier-free guidance 则训练同一个 denoiser 同时处理有条件和无条件输入，生成时比较两种预测。

若无条件预测为 epsilon_uncond，有条件预测为 epsilon_cond，常见 CFG 写法是：

~~~math
\hat{\epsilon}_{\mathrm{cfg}}
=
\epsilon_{\mathrm{uncond}}
+
w
\left(
\epsilon_{\mathrm{cond}}
-
\epsilon_{\mathrm{uncond}}
\right).
~~~

这里 w 是 guidance scale。按这个写法：

- w=0 是无条件预测；
- w=1 等于有条件预测；
- w>1 放大条件方向。

有些库会把 guidance scale 定义成另一种偏移后的参数，阅读实现时不能只看变量名，必须看实际公式。

### 8.3 条件 dropout 的作用

训练 CFG 时，部分样本会把条件置空或替换成 null condition。这样同一个网络学到条件和无条件两种预测。条件 dropout 的比例影响：

- 无条件分支的质量；
- 条件分支的稳定性；
- CFG 可用范围；
- 训练数据的有效条件分布。

如果训练时从未出现空条件，推理时再强行调用无条件分支，通常没有可靠依据。

### 8.4 guidance 的质量和多样性折中

提高 w 往往会让结果更贴近条件方向，但也可能：

- 颜色过饱和；
- 局部结构失真；
- 多个对象被错误合并；
- 纹理变得不自然；
- 不同 seed 的多样性下降。

因此 guidance scale 不是越大越好。评估至少要同时看 prompt 遵循、图像质量、多样性和失败率，并固定采样器、步数和 seed 集合。

### 8.5 Negative prompt 的边界

negative prompt 可以被视为另一个条件输入，或某些产品实现中的条件约束。它不是一个能自动删除所有不想要内容的“反向魔法”。negative prompt 的效果依赖 text encoder、训练数据、CFG 公式和采样器。

本章只把它放在条件注入的边界上。具体产品模板、权重组合和图像编辑接口，应在对应系统章节中按版本核对。

## 9. Latent diffusion：在压缩空间中扩散

### 9.1 像素空间的成本

若图片有 H×W 个空间位置、C 个通道，像素状态元素数量是：

~~~math
N_{\mathrm{pixel}}=HWC.
~~~

每一个采样步都要在这个状态上运行 denoiser。高分辨率会增加卷积、激活和注意力成本。

### 9.2 VAE 与 latent 状态

Latent diffusion 先用 VAE encoder 把图片映射为 latent z：

~~~math
z=E_{\phi}(x),
\qquad
\hat{x}=D_{\psi}(z).
~~~

diffusion 在 z 上进行：

~~~math
z_t=a_t z_0+s_t\epsilon.
~~~

生成结束后通过 decoder 得到图片：

~~~text
image
    -> VAE encoder
    -> latent diffusion
    -> VAE decoder
    -> image
~~~

VAE 不是透明的无损搬运。压缩会影响纹理、小字、边缘和色彩；decoder 会引入自己的重建误差。latent diffusion 的质量应同时评估 latent denoiser 和 VAE 重建。

### 9.3 空间压缩的数量级

若 VAE 把高和宽都压缩 f 倍，latent 空间位置数大约是原来的：

~~~math
R_{\mathrm{space}}
=
\frac{H/f\cdot W/f}{H\cdot W}
=
\frac{1}{f^2}.
~~~

如果像素通道为 C_pixel，latent 通道为 C_latent，则仅按状态元素估算：

~~~math
R_{\mathrm{state}}
\approx
\frac{C_{\mathrm{latent}}}{C_{\mathrm{pixel}}f^2}.
~~~

例如 f=8、C_pixel=3、C_latent=4：

~~~math
R_{\mathrm{state}}
\approx
\frac{4}{3\times64}
=
\frac{1}{48}
\approx0.0208.
~~~

这是状态元素的教学估算，不是端到端延迟比。VAE、denoiser 宽度、attention、精度和采样步数仍会决定实际成本。

### 9.4 latent scaling 与实现契约

一些系统会在送入 denoiser 前对 latent 乘一个缩放因子，或在 decoder 前做相反变换。这个 scaling 是实现契约的一部分：

- 训练数据构造时怎样编码；
- scheduler 认为 latent 的方差范围是什么；
- denoiser 训练时看到的数值尺度是什么；
- decoder 需要怎样反缩放。

如果只替换 VAE 而保留旧 denoiser 和 scheduler，latent 分布可能不匹配。检查模型时要把 VAE、scaling factor、归一化和 dtype 一起记录。

## 10. 从少步采样到相邻生成路线

### 10.1 Progressive distillation

标准 diffusion 可能需要很多 denoiser evaluations。蒸馏方法用一个较慢的 teacher 轨迹生成监督，让 student 用更少步骤近似 teacher 的结果。

蒸馏的代价是：

- 需要额外 teacher 计算；
- student 可能只适应特定步数或 sampler；
- 少步误差会集中到结构、文字和细节；
- teacher 的偏差可能被继承。

因此“4 步模型”并不是凭空出现的，它通常把多步路径的行为压缩进另一个训练过程。

### 10.2 Consistency models

Consistency 路线希望不同噪声水平沿同一轨迹映射到一致的 clean endpoint，因而可以支持较少步甚至一步的生成。它和 DDPM 的训练目标、网络输出和采样方式不完全相同。

阅读 consistency 实现时，要问：

- consistency 的时间对和 target 如何构造；
- teacher/student 或 EMA 如何使用；
- 一个步和多步的推理公式是什么；
- 训练的边界时间和噪声分布是什么。

不能因为结果看起来也从噪声变图片，就把 consistency loss 当作 epsilon MSE 的别名。

### 10.3 Flow matching

Flow matching 直接学习一条概率路径上的速度场。一个简单的线性插值例子是：

~~~math
x_t=(1-t)x_0+t x_1,
\qquad
u_t=x_1-x_0.
~~~

模型学习：

~~~math
v_\theta(x_t,t,c)\approx u_t.
~~~

当 x_1 取噪声端点时，这条路线与从噪声到数据的连续运输有关；但它的训练路径、目标和 solver 需要单独定义。Flow matching、DDPM 和 score SDE 可以在连续生成的讨论中相互联系，却不能只通过“都用了 t”来互换公式。

### 10.4 如何选择路线

如果目标是建立基本推理能力，先掌握 DDPM 的闭式加噪、epsilon 目标和反向均值；如果目标是理解连续采样，再学习 score/SDE 和 ODE；如果目标是少步部署，再研究 solver、蒸馏、consistency 或 flow matching。

工程选择要根据：

- 训练成本；
- 采样步数；
- 单步 denoiser 成本；
- 条件遵循；
- 细节和文字质量；
- 多样性；
- 目标硬件；
- 是否要支持编辑、控制或视频。

## 11. 一个文本到图像系统的完整数据流

### 11.1 训练路径

一条简化训练路径是：

~~~text
image x0 + text c
    -> VAE encode or pixel normalization
    -> sample t and epsilon
    -> construct xt
    -> text encoder(c)
    -> denoiser(xt, t, text condition)
    -> epsilon / v / x0 target
    -> loss and optimizer
~~~

如果训练使用条件 dropout，还要记录哪些样本被置空条件。若使用 latent diffusion，还要记录 VAE 的版本和 scaling。

### 11.2 推理路径

~~~text
prompt
    -> tokenizer and text encoder
    -> initial noise or latent
    -> scheduler timesteps
    -> denoiser and optional CFG at each step
    -> final latent
    -> VAE decode
    -> safety and post-processing checks
~~~

这里的“安全检查”不能被理解为 diffusion 数学本身已经解决了内容安全。它是系统层的独立组件，可能包括输入策略、输出分类、人工复核、版权或隐私策略。

### 11.3 训练和推理最容易不一致的地方

- 训练使用 epsilon，推理 scheduler 按 v 解释；
- 训练 latent 有 scaling，推理忘记 scaling；
- 训练文本 encoder 版本与推理不同；
- 训练条件 dropout，推理 unconditional 分支形状错误；
- 训练分辨率和推理分辨率的位置表示不兼容；
- 训练 beta schedule 与推理时间网格不一致；
- 训练使用随机 crop，推理使用不同归一化。

这些问题常常不会在代码启动时立刻报错，却会表现为 prompt 遵循下降、结构不稳定或采样质量异常。

## 12. 常见失败模式与实验诊断

### 12.1 输出几乎全黑或全白

可能原因：

- latent scaling 错；
- VAE 归一化范围错；
- sampler 的 sigma 或 alpha 索引错；
- dtype 下溢或溢出；
- decoder 输入范围不对。

最小实验是固定一个 latent，分别绕过 denoiser 检查 VAE encode/decode，再打印每个时间步的均值、方差和最大绝对值。

### 12.2 画面有主题但 prompt 遵循差

可能原因：

- text encoder 没有正确连接；
- cross-attention mask 或条件顺序错；
- CFG 的 unconditional 和 conditional 绑定反了；
- guidance scale 不适合当前模型；
- 训练 caption 太粗或条件 dropout 过强。

可以固定 seed，分别用空 prompt、短 prompt、关键词交换和不同 guidance 做配对实验。

### 12.3 文字和数字乱码

可能原因：

- 训练数据中清晰文字样本不足；
- latent 压缩丢失高频细节；
- denoiser 没有足够的文字结构监督；
- 采样步数或 guidance 把局部结构推坏；
- 目标本身超出模型分辨率。

不能用“整体图像看起来不错”替代文字专项评估。应单独记录文字可读率、字符错误率和不同字号切片。

### 12.4 结构重复、模式坍缩与训练分布

如果不同随机 seed 生成的图片几乎相同，或大量 prompt 都得到相同构图，问题可能不在 sampler 一处：训练数据可能过度重复，条件覆盖过窄，模型过拟合某些布局，或者 guidance 把不同轨迹都推向同一方向。对 diffusion 来说，这不一定表现为 GAN 语境下的经典模式坍缩，但结果层面同样会表现为多样性下降。

诊断时可以固定 prompt，采样多个 seed，记录感知相似度、对象数量和构图位置；再固定 seed，交换 prompt 中的关键条件。若 seed 改变而输出几乎不变，检查初始噪声是否真的进入 sampler；若 prompt 改变而输出不变，检查条件编码、cross-attention 和训练数据覆盖。质量提高但多样性降低时，还要单独比较 guidance scale，而不能只看单张样例。

### 12.5 手指、人体或多物体关系异常

可能原因：

- 长尾结构数据不足；
- patch 或 latent 表示丢失局部关系；
- 多对象组合在训练中不充分；
- 高 guidance 放大错误方向；
- 少步采样误差过大。

配对测试应固定 prompt，只改变步数、guidance、seed 和分辨率中的一个变量，否则无法归因。

### 12.6 采样加速后质量突然下降

可能原因：

- 推理时间步过少；
- solver 与 prediction type 不匹配；
- teacher 蒸馏范围与目标分布不一致；
- 使用了不适配的 sigma schedule；
- VAE 或后处理成为新的瓶颈。

比较少步模型时要报告 NFE、总延迟、峰值显存、质量、多样性和失败率，而不是只报告步数。

### 12.7 训练 loss 正常但生成失败

loss 正常只能说明训练批次上的数值目标在下降。还可能存在：

- label 与 prediction shift 错；
- 训练样本的 x0 范围与推理初始分布不一致；
- 条件没有真正进入网络；
- 训练 VAE 与推理 VAE 不同；
- 只在容易时间步上优化；
- 采样器使用了错误的 reverse formula。

因此应保存固定小批次的 xt、target、prediction、x0_hat 和一条完整采样轨迹，做端到端回归。

## 13. 最小可运行 diffusion 算术审计

下面的程序只使用 Python 标准库，用四维向量模拟核心公式。它不生成图片，也不代表真实 denoiser 的能力；它要检查的是：

- alpha_bar 是否随时间步递减；
- 闭式加噪是否保持线性关系；
- 噪声 MSE 是否按预期计算；
- epsilon prediction 能否近似恢复 x0；
- reverse mean 是否使用同一套 schedule；
- score 与 epsilon 的比例关系；
- v prediction 的互相转换；
- CFG 是否真的沿条件方向移动；
- latent 状态元素的数量级是否变小。

~~~python
import math


def round_list(values, ndigits=4):
    return [round(value, ndigits) for value in values]


def require_finite_vector(values, name):
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")


def validate_same_length(*named_vectors):
    lengths = {len(values) for _, values in named_vectors}
    if len(lengths) != 1:
        names = ", ".join(name for name, _ in named_vectors)
        raise ValueError(f"vectors must have the same length: {names}")


def validate_schedule(betas):
    if not betas:
        raise ValueError("schedule must contain at least one beta")
    if not all(math.isfinite(beta) and 0.0 < beta < 1.0 for beta in betas):
        raise ValueError("each beta must be finite and in (0, 1)")
    alphas = [1.0 - beta for beta in betas]
    alpha_bars = []
    running = 1.0
    for alpha in alphas:
        running *= alpha
        if not 0.0 < running < 1.0:
            raise ValueError("alpha_bar must stay in (0, 1)")
        alpha_bars.append(running)
    return alphas, alpha_bars


def validate_timestep(t, total_steps):
    if isinstance(t, bool) or not isinstance(t, int):
        raise ValueError("timestep must be an integer")
    if not 1 <= t <= total_steps:
        raise ValueError("timestep must be in 1..T")


def reverse_mean_from_epsilon(noisy, beta, alpha, alpha_bar, predicted):
    if not 0.0 < beta < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("beta and alpha must be in (0, 1)")
    if not 0.0 < alpha_bar < 1.0:
        raise ValueError("alpha_bar must be in (0, 1)")
    require_finite_vector(noisy, "noisy")
    require_finite_vector(predicted, "predicted noise")
    validate_same_length(("noisy", noisy), ("predicted", predicted))
    scale = math.sqrt(alpha)
    noise_scale = math.sqrt(1.0 - alpha_bar)
    return [
        (value - beta / noise_scale * estimate) / scale
        for value, estimate in zip(noisy, predicted)
    ]


def posterior_mean(noisy, clean, beta, alpha, alpha_bar, previous_alpha_bar):
    if not 0.0 < beta < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("beta and alpha must be in (0, 1)")
    if not 0.0 < alpha_bar < 1.0 or not 0.0 <= previous_alpha_bar <= 1.0:
        raise ValueError("alpha_bar values are outside their domain")
    require_finite_vector(noisy, "noisy")
    require_finite_vector(clean, "clean")
    validate_same_length(("noisy", noisy), ("clean", clean))
    denominator = 1.0 - alpha_bar
    return [
        (
            math.sqrt(previous_alpha_bar) * beta / denominator * clean_value
            + math.sqrt(alpha) * (1.0 - previous_alpha_bar) / denominator * noisy_value
        )
        for noisy_value, clean_value in zip(noisy, clean)
    ]


def cfg_combine(uncond, cond, guidance_scale):
    if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
        raise ValueError("guidance scale must be finite and non-negative")
    require_finite_vector(uncond, "unconditional prediction")
    require_finite_vector(cond, "conditional prediction")
    validate_same_length(("unconditional", uncond), ("conditional", cond))
    return [
        unconditional + guidance_scale * (conditional - unconditional)
        for unconditional, conditional in zip(uncond, cond)
    ]


def latent_state_ratio(height, width, pixel_channels, factor, latent_channels):
    if min(height, width, pixel_channels, factor, latent_channels) <= 0:
        raise ValueError("latent dimensions and compression settings must be positive")
    if height % factor or width % factor:
        raise ValueError("height and width must be divisible by the factor")
    pixel_values = height * width * pixel_channels
    latent_values = (height // factor) * (width // factor) * latent_channels
    return latent_values / pixel_values, latent_values, pixel_values


x0 = [0.2, -0.4, 0.7, -0.1]
noise = [0.5, -1.0, 0.25, 0.75]
pred_noise = [0.45, -0.9, 0.3, 0.7]
betas = [0.0001, 0.005, 0.01, 0.015, 0.02]

require_finite_vector(x0, "x0")
require_finite_vector(noise, "noise")
require_finite_vector(pred_noise, "predicted noise")
validate_same_length(
    ("x0", x0),
    ("noise", noise),
    ("predicted noise", pred_noise),
)
alphas, alpha_bars = validate_schedule(betas)

t = 4
validate_timestep(t, len(betas))
index = t - 1
alpha_t = alphas[index]
alpha_bar_t = alpha_bars[index]
a_t = math.sqrt(alpha_bar_t)
s_t = math.sqrt(1.0 - alpha_bar_t)

x_t = [
    a_t * clean + s_t * eps
    for clean, eps in zip(x0, noise)
]

noise_mse = sum(
    (pred - eps) ** 2 for pred, eps in zip(pred_noise, noise)
) / len(noise)

x0_hat = [
    (noisy - s_t * pred) / a_t
    for noisy, pred in zip(x_t, pred_noise)
]
reconstruction_mae = sum(
    abs(recon - clean) for recon, clean in zip(x0_hat, x0)
) / len(x0)

beta_t = betas[index]
ddpm_mean = [
    value
    for value in reverse_mean_from_epsilon(
        x_t,
        beta_t,
        alpha_t,
        alpha_bar_t,
        pred_noise,
    )
]

previous_alpha_bar = 1.0 if t == 1 else alpha_bars[index - 1]
posterior_from_x0 = posterior_mean(
    x_t,
    x0_hat,
    beta_t,
    alpha_t,
    alpha_bar_t,
    previous_alpha_bar,
)
posterior_consistency_error = max(
    abs(from_epsilon - from_x0)
    for from_epsilon, from_x0 in zip(ddpm_mean, posterior_from_x0)
)
first_step_posterior = posterior_mean(
    x_t,
    x0_hat,
    betas[0],
    alphas[0],
    alpha_bars[0],
    1.0,
)

score = [-eps / s_t for eps in noise]

v_target = [
    a_t * eps - s_t * clean
    for clean, eps in zip(x0, noise)
]
x0_from_v = [
    a_t * noisy - s_t * velocity
    for noisy, velocity in zip(x_t, v_target)
]
eps_from_v = [
    s_t * noisy + a_t * velocity
    for noisy, velocity in zip(x_t, v_target)
]
v_roundtrip_error = max(
    abs(recovered - clean)
    for recovered, clean in zip(x0_from_v, x0)
)
eps_roundtrip_error = max(
    abs(recovered - eps)
    for recovered, eps in zip(eps_from_v, noise)
)

uncond_eps = [0.7, -0.8, 0.2, 0.5]
cond_eps = [0.4, -1.1, 0.3, 0.8]
guidance_scale = 3.0
cfg_guided_eps = cfg_combine(uncond_eps, cond_eps, guidance_scale)

latent_ratio, latent_values, pixel_values = latent_state_ratio(
    height=512,
    width=512,
    pixel_channels=3,
    factor=8,
    latent_channels=4,
)

checks = {
    "alpha_bar_decreases": all(
        alpha_bars[i] > alpha_bars[i + 1]
        for i in range(len(alpha_bars) - 1)
    ),
    "xt_shape_ok": len(x_t) == len(x0),
    "mse_positive": noise_mse > 0,
    "reconstruction_close": reconstruction_mae < 0.02,
    "v_roundtrip_ok": v_roundtrip_error < 1e-12,
    "epsilon_roundtrip_ok": eps_roundtrip_error < 1e-12,
    "posterior_mean_matches": posterior_consistency_error < 1e-12,
    "first_step_posterior_defined": len(first_step_posterior) == len(x0),
    "cfg_changes_prediction": (
        cfg_guided_eps != cond_eps
        and cfg_guided_eps != uncond_eps
    ),
    "latent_is_smaller": latent_ratio < 0.05,
}


def expects_value_error(function):
    try:
        function()
    except ValueError:
        return True
    return False


boundary_checks = {
    "invalid_beta_rejected": expects_value_error(
        lambda: validate_schedule([0.0, 0.1])
    ),
    "out_of_range_timestep_rejected": expects_value_error(
        lambda: validate_timestep(0, len(betas))
    ),
    "negative_previous_alpha_bar_rejected": expects_value_error(
        lambda: posterior_mean(
            x_t,
            x0_hat,
            betas[0],
            alphas[0],
            alpha_bars[0],
            -0.1,
        )
    ),
    "empty_vector_rejected": expects_value_error(
        lambda: require_finite_vector([], "empty")
    ),
    "mismatched_vector_rejected": expects_value_error(
        lambda: validate_same_length(("a", [1.0]), ("b", [1.0, 2.0]))
    ),
    "mismatched_cfg_vector_rejected": expects_value_error(
        lambda: cfg_combine([0.0], [1.0, 2.0], 1.0)
    ),
    "non_divisible_latent_rejected": expects_value_error(
        lambda: latent_state_ratio(511, 512, 3, 8, 4)
    ),
}

print("alpha_bars=", round_list(alpha_bars, 6))
print("t=", t, "alpha_bar_t=", round(alpha_bar_t, 6))
print("x_t=", round_list(x_t, 4))
print("noise_mse=", round(noise_mse, 5))
print("x0_hat=", round_list(x0_hat, 4))
print("reconstruction_mae=", round(reconstruction_mae, 5))
print("ddpm_mean=", round_list(ddpm_mean, 4))
print("posterior_consistency_error=", posterior_consistency_error)
print("score=", round_list(score, 4))
print("v_roundtrip_error=", v_roundtrip_error)
print("eps_roundtrip_error=", eps_roundtrip_error)
print("cfg_guided_eps=", round_list(cfg_guided_eps, 4))
print(
    "latent_values=",
    latent_values,
    "pixel_values=",
    pixel_values,
    "latent_ratio=",
    round(latent_ratio, 4),
)
print("sampling_calls=", {"ddpm_50": 50, "ddim_20": 20, "distilled_4": 4})
print("checks=", checks)
print("boundary_checks=", boundary_checks)
print(
    "all_checks_passed=",
    all(checks.values()) and all(boundary_checks.values()),
)
~~~

这段程序会输出类似：

~~~text
alpha_bars= [0.9999, 0.9949, 0.984951, 0.970177, 0.950774]
t= 4 alpha_bar_t= 0.970177
x_t= [0.2833, -0.5667, 0.7327, 0.031]
noise_mse= 0.00437
x0_hat= [0.2088, -0.4175, 0.6912, -0.0912]
reconstruction_mae= 0.01096
ddpm_mean= [0.2461, -0.4922, 0.712, -0.03]
posterior_consistency_error= 1.6653345369377348e-15
score= [-2.8953, 5.7906, -1.4477, -4.343]
v_roundtrip_error= 1.1102230246251565e-16
eps_roundtrip_error= 2.220446049250313e-16
cfg_guided_eps= [-0.2, -1.7, 0.5, 1.4]
latent_values= 16384 pixel_values= 786432 latent_ratio= 0.0208
sampling_calls= {'ddpm_50': 50, 'ddim_20': 20, 'distilled_4': 4}
checks= {'alpha_bar_decreases': True, 'xt_shape_ok': True, 'mse_positive': True, 'reconstruction_close': True, 'v_roundtrip_ok': True, 'epsilon_roundtrip_ok': True, 'posterior_mean_matches': True, 'cfg_changes_prediction': True, 'latent_is_smaller': True}
boundary_checks= {'invalid_beta_rejected': True, 'out_of_range_timestep_rejected': True, 'negative_previous_alpha_bar_rejected': True, 'empty_vector_rejected': True, 'mismatched_vector_rejected': True, 'mismatched_cfg_vector_rejected': True, 'non_divisible_latent_rejected': True}
all_checks_passed= True
~~~

这些 checks 只证明四维教学算术彼此一致，并验证非法 schedule、时间步、向量、CFG 形状和 latent 尺寸不会静默通过。它们不证明网络预测准确，也不证明采样器在真实图片上有效。尤其是 latent_ratio 只按状态元素估算，不能当成端到端延迟或显存比例。

## 14. 练习：从公式到采样系统

1. 解释为什么 forward process 可以固定，而 reverse process 需要学习。
2. 给定 beta_1、beta_2 和一个 x0，手算 alpha、bar alpha 和闭式 xt。
3. 说明为什么随机抽取一个时间步就能训练，而不必每次走完整条 forward 链。
4. 从 xt=a_t x0+s_t epsilon 推导 x0_hat。
5. 推导 v prediction 与 x0、epsilon 的互相转换。
6. 解释条件 forward score 与边缘分布 score 的区别。
7. 写出 DDPM 反向均值，并说明 beta_t、alpha_t、bar alpha_t 必须来自同一 schedule。
8. 比较 DDPM、DDIM 和 ODE 风格采样的随机性与成本。
9. 计算一个 1024×1024、3 通道像素状态和压缩 8 倍、4 通道 latent 状态的元素比例。
10. 说明 U-Net 的 skip connection、time embedding 和 cross-attention 各自承担什么责任。
11. 计算 latent 高宽为 64、patch size 为 2 时 DiT 的 token 数，并讨论注意力成本。
12. 用一个数值例子验证 CFG 公式中的加号和 guidance scale。
13. 设计一个实验区分 text encoder 失效、cross-attention 失效和 guidance 过强。
14. 解释为什么少步蒸馏模型需要独立验证，不能只看采样步数。
15. 比较 consistency 和 flow matching 与 DDPM epsilon prediction 的训练对象。
16. 为一次生成失败建立记录：固定 seed、prompt、scheduler、步数、guidance、VAE 和硬件，并设计最小消融。

## 15. 资料入口与证据边界

1. Ho et al., Denoising Diffusion Probabilistic Models：<https://arxiv.org/abs/2006.11239>。支持离散 forward/reverse diffusion、DDPM 训练和采样背景。
2. Song et al., Denoising Diffusion Implicit Models：<https://arxiv.org/abs/2010.02502>。支持 DDIM 采样路径和确定性/随机性讨论。
3. Song et al., Score-Based Generative Modeling through Stochastic Differential Equations：<https://arxiv.org/abs/2011.13456>。支持 score、SDE、reverse-time SDE 和 probability flow ODE 视角。
4. Dhariwal and Nichol, Diffusion Models Beat GANs on Image Synthesis：<https://arxiv.org/abs/2105.05233>。支持 classifier guidance 研究背景。
5. Ho and Salimans, Classifier-Free Diffusion Guidance：<https://arxiv.org/abs/2207.12598>。支持条件 dropout、无条件/有条件预测和 CFG 构造。
6. Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models：<https://arxiv.org/abs/2112.10752>。支持 VAE latent 空间扩散、cross-attention 条件和成本动机。
7. Peebles and Xie, Scalable Diffusion Models with Transformers：<https://arxiv.org/abs/2212.09748>。支持 DiT 作为 diffusion denoiser 的研究路线。
8. Lu et al., DPM-Solver：<https://arxiv.org/abs/2206.00927>。支持少步数值求解器的研究背景，不能保证任意模型直接适配。
9. Salimans and Ho, Progressive Distillation for Fast Sampling of Diffusion Models：<https://arxiv.org/abs/2202.00512>。支持把多步采样行为蒸馏到更少步的研究路线。
10. Song et al., Consistency Models：<https://arxiv.org/abs/2303.01469>。支持 consistency 生成路线，不能把其目标当成 DDPM MSE。
11. Lipman et al., Flow Matching for Generative Modeling：<https://arxiv.org/abs/2210.02747>。支持速度场和概率路径的 flow matching 研究背景。
12. Hugging Face Diffusers scheduler 文档：<https://huggingface.co/docs/diffusers/using-diffusers/schedulers>。支持特定版本 scheduler 的接口说明；实际默认值、prediction type 和 scaling 仍要绑定安装版本。

原始论文适合支持方法定义和论文实验，官方文档适合支持 API 行为，教学 demo 只支持公式算术。上述资料不能单独证明某个产品的图片质量、prompt 遵循、文字可读率、版权安全或单位成功任务成本；这些结论必须在目标系统和目标硬件上复测。

## 16. 本章回顾

Diffusion 的主线可以压缩为：

~~~text
clean sample
    -> prescribed forward noise path
    -> noisy state at timestep t
    -> denoiser predicts epsilon / x0 / v / score
    -> scheduler or solver updates the state
    -> generated sample
~~~

DDPM 用离散高斯链建立最清楚的入口：alpha 和 bar alpha 给出闭式加噪，epsilon prediction 提供可计算监督，反向均值把网络预测变成采样更新。score/SDE 视角把这条路径推广到连续时间；DDIM、Euler、Heun 和 DPM-Solver 改变采样轨迹或数值积分；蒸馏和 consistency 试图减少函数评估；flow matching 学习另一类概率路径上的速度场。

U-Net 和 DiT 是 denoiser 的不同结构选择；文本通过 cross-attention 或其他条件路径影响去噪；CFG 用有条件和无条件预测的差值增强条件方向。Latent diffusion 通过 VAE 降低状态空间的空间成本，却把 VAE 重建、latent scaling 和 decoder 误差加入系统契约。

真正可靠的 diffusion 工程，不是记住“加噪、去噪、CFG、latent”几个名词，而是能把训练 schedule、prediction type、时间网格、采样器、条件路径、VAE、精度、步数和评估指标放在同一张账本里。下一章将进入 Stable Diffusion 与 DALL·E，具体讨论完整文生图系统如何组织 VAE、文本编码器、denoiser、prompt、控制和图像编辑。
