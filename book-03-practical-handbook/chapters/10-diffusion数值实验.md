# 第 10 章 Diffusion 数值实验

本章把扩散模型的前向加噪和反向去噪写成可复算的小实验。我们跟踪少量标量和向量，确认调度系数、噪声目标与一次反向更新如何对应；它是理解机制的实验台，不是完整文生图系统，也不代表真实图像质量。更完整的生成任务和 pipeline 由第十五册深入。

本章沿用原第 68 讲编号。

## 第 68 讲：实现 Diffusion 加噪与去噪最小例子

### 68.1 把生成拆成一条随机过程

扩散模型的核心不是“画图 API”，而是把难以直接建模的数据分布拆成许多步简单的去噪任务。正向过程从真实数据 x_0 出发逐步加噪，反向过程从近似高斯噪声开始逐步恢复数据。

DDPM 的一步正向过程为：

~~~math
q(x_t\mid x_{t-1})=\mathcal{N}\left(\sqrt{1-\beta_t}x_{t-1},\beta_tI\right)
~~~

定义：

~~~math
\alpha_t=1-\beta_t,\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
~~~

标准 DDPM schedule 通常要求每个 \(0<\beta_t<1\)，因此 \(0<\alpha_t<1\)，并按合法的时间步顺序计算 \(\bar{\alpha}_t\)。若 \(\beta_t\) 越界，平方根、后验方差和采样过程都不再满足这里的概率解释。

可以直接从 x_0 构造任意时刻的 x_t：

~~~math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad\epsilon\sim\mathcal{N}(0,I)
~~~

累计信噪比可以用：

~~~math
\mathrm{SNR}_t=\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
~~~

该有限 SNR 定义要求 \(0<\bar{\alpha}_t<1\)。干净边界约定 \(\bar{\alpha}_0=1\)，其 SNR 在扩展意义上为正无穷；这不是非法的干净样本，只是不能代入要求有限分母的噪声反解。代码中的 `step=0` 对应数学上的 `t=1`，即第一次加噪。

随着 t 增大，信号占比下降。schedule 的本质是设计信噪比如何变化，而不是随便生成一串 beta。

### 68.2 为什么通常预测噪声

训练时可以采样 x_0、t 和真实噪声 epsilon，再构造 x_t。模型学习：

~~~math
\hat{\epsilon}=\epsilon_\theta(x_t,t)
~~~

单个训练样本与时间步的简化损失是：

~~~math
L_{\mathrm{simple}}=\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2
~~~

完整训练目标还要对数据、采样时间步与高斯噪声求期望；实现中采用 sum 还是按元素 mean 会改变尺度，需保持一致。

如果已知噪声预测，可以构造反向均值：

~~~math
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}\right)
~~~

再根据选择的方差注入随机性。后验方差的一种写法为：

~~~math
\tilde{\beta}_t=\beta_t\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
~~~

这里取 \(\bar{\alpha}_0=1\)，因此最后一次反向更新 `t=1` 的后验方差为零。若另选固定方差，DDPM 采样算法仍在 `t=1` 使用零随机项。预测 x_0、预测 epsilon 和预测 velocity 是不同参数化，训练目标与采样公式必须匹配。

### 68.3 纯 Python 公式审计

> 代码状态：已验证（2026-10-22；Python 3.12.9，标准库教学输入；历史记录中的输入、输出与边界检查适用于此代码）

~~~python
import math


betas = [0.01, 0.02, 0.03, 0.04, 0.05]
if (not betas or any(isinstance(beta, bool) or not isinstance(beta, (int, float))
                     or not math.isfinite(beta) or not 0 < beta < 1
                     for beta in betas)):
    raise ValueError("betas must be finite values in (0, 1)")
alphas = [1.0 - beta for beta in betas]
alpha_bars = []
product = 1.0
for alpha in alphas:
    product *= alpha
    alpha_bars.append(product)


def q_sample(x0, noise, step):
    if (not isinstance(x0, list) or not isinstance(noise, list)
            or not x0 or len(x0) != len(noise)):
        raise ValueError("x0 and noise must be non-empty and aligned")
    if (isinstance(step, bool) or not isinstance(step, int)
            or not 0 <= step < len(alpha_bars)):
        raise ValueError("step is out of range")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in x0 + noise):
        raise ValueError("sample values must be finite numbers")
    alpha_bar = alpha_bars[step]
    return [
        math.sqrt(alpha_bar) * value
        + math.sqrt(1.0 - alpha_bar) * eps
        for value, eps in zip(x0, noise)
    ]


def recover_noise(x_t, x0, step):
    if (not isinstance(x_t, list) or not isinstance(x0, list)
            or not x_t or len(x_t) != len(x0)):
        raise ValueError("x_t and x0 must be non-empty and aligned")
    if (isinstance(step, bool) or not isinstance(step, int)
            or not 0 <= step < len(alpha_bars)):
        raise ValueError("step is out of range")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in x_t + x0):
        raise ValueError("sample values must be finite numbers")
    alpha_bar = alpha_bars[step]
    if alpha_bar >= 1.0:
        raise ValueError("step must contain non-zero noise")
    return [
        (current - math.sqrt(alpha_bar) * clean)
        / math.sqrt(1.0 - alpha_bar)
        for current, clean in zip(x_t, x0)
    ]


def reverse_step(x_t, predicted_noise, step, random_noise):
    if (isinstance(step, bool) or not isinstance(step, int)
            or not 0 <= step < len(betas)):
        raise ValueError("step is out of range")
    vectors = (x_t, predicted_noise, random_noise)
    if any(not isinstance(v, list) or not v for v in vectors):
        raise ValueError("vectors must be non-empty lists")
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("vector dimensions must match")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for vector in vectors for v in vector):
        raise ValueError("vector values must be finite numbers")
    alpha, alpha_bar = alphas[step], alpha_bars[step]
    previous_bar = 1.0 if step == 0 else alpha_bars[step - 1]
    variance = betas[step] * (1 - previous_bar) / (1 - alpha_bar)
    mean = [(current - betas[step] * eps / math.sqrt(1 - alpha_bar))
            / math.sqrt(alpha) for current, eps in zip(x_t, predicted_noise)]
    return [value + math.sqrt(variance) * z
            for value, z in zip(mean, random_noise)], variance


x0 = [2.0, 0.0]
noise = [0.3, -1.2]
x_t = q_sample(x0, noise, 4)
recovered = recover_noise(x_t, x0, 4)
mse = sum((a - b) ** 2 for a, b in zip(recovered, noise)) / len(noise)
first_noisy = q_sample(x0, noise, 0)
last_result, last_variance = reverse_step(first_noisy, noise, 0, [100.0, -100.0])
assert last_variance == 0
assert all(abs(a - b) < 1e-12 for a, b in zip(last_result, x0))
previous, variance = reverse_step(x_t, noise, 4, [0.0, 0.0])

assert alpha_bars[-1] < alpha_bars[0]
assert mse < 1e-12
print("alpha_bars=", [round(value, 6) for value in alpha_bars])
print("x_t=", [round(value, 4) for value in x_t])
print("recovered_noise=", [round(value, 4) for value in recovered])
print("recovery_mse=", round(mse, 12))
print("reverse_mean=", [round(value, 4) for value in previous])
print("posterior_variance=", round(variance, 6))
~~~
前向输出约 `[1.9658,-0.4518]`，反解噪声 `[0.3,-1.2]`；反向函数用指定噪声预测计算一步均值与后验方差，随机项可由调用方用标准高斯生成。`step=0` 时不论传入什么有限随机项，方差均为零，使用真实噪声预测可回到原样本。其他步的后验仍有随机性，单步均值不等于精确恢复实际前一状态。

这里的 `recover_noise` 直接使用干净 `x0`，是代数一致性检查，不能充当生成时可用的噪声预测器。五步 schedule 的最终累计 alpha 仍约 0.8583，远未达到近似纯噪声；它不是可直接从标准高斯开始生成的训练配置。实验不训练网络，也不证明采样器能恢复目标分布；完整 toy DDPM 需要另训练噪声预测器并比较数据与采样分布。

### 68.4 从二维 toy 到图像系统

二维点的 MLP 在图像系统中通常会替换成 UNet 或 DiT；输入从 [batch,2] 变成像素或 latent 张量；时间步 embedding 仍然告诉模型当前噪声水平。Latent Diffusion 先使用 VAE 将图片压缩到 latent 空间，再在 latent 上做扩散：

~~~text
image -> VAE encoder -> latent z
latent z -> conditional diffusion -> generated latent
generated latent -> VAE decoder -> image
~~~

latent 空间减少空间尺寸和计算量，但 VAE 会带来重建损失。图像生成系统的质量因此同时受数据、VAE、去噪网络、条件编码、采样步数和解码器影响。

### 68.5 文本条件与 classifier-free guidance

有文本条件时，模型变成：

~~~math
\hat{\epsilon}=\epsilon_\theta(x_t,t,c)
~~~

Classifier-free guidance 同时计算有条件和无条件预测：

~~~math
\epsilon_{\mathrm{cfg}}
=\epsilon_{\mathrm{uncond}}
+w(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}})
~~~

w 增大通常会增强 prompt adherence，但也可能带来过饱和、构图僵硬和细节崩坏。它不是“越大越好”的质量旋钮，应在提示词类型、图片风格和采样器上分桶比较。

DDIM、DPM-Solver、consistency model 和蒸馏方法主要解决采样步数与质量的 trade-off；它们改变采样路径或模型训练方式，并不等于基础 DDPM 的 forward process 消失。

#### 跟着算一次：前向信号如何变小，反向均值如何算

取一个只有两个时间步的教学 schedule：β1=0.1、β2=0.2，因此 α1=0.9、α2=0.8，累计值分别是 ᾱ1=0.9、ᾱ2=0.72。令干净标量 x0=1，抽到的噪声 ε=-0.5。第二步的带噪样本为 √0.72×1+√0.28×(-0.5)，约为 0.5839。第一项信号贡献约 0.8485，第二项噪声贡献约 -0.2646；观察到的数值是两项叠加，不是模型从图像里“减掉一点噪声”这么简单。

如果真实噪声预测器恰好给出 ε̂=-0.5，第二步反向均值为 (0.5839-0.2×(-0.5)/√0.28)/√0.8，约 0.7729。后验方差为 0.2×(1-0.9)/(1-0.72)，约 0.0714。若按该后验采样，标准差约 0.2672，再乘一个独立标准高斯随机数。一次真实反向抽样因而不必等于当初正向路径上的 x1；反向分布表达的是给定 xt 后对前一状态的不确定性。

这个两步 schedule 的最终 ᾱ 仍是 0.72，第二步样本里还有很强的原始信号。真实生成通常从约定的初始噪声分布开始，不能把这个短 schedule 的输出冒充成“已经纯噪声”。更重要的是，上面的噪声预测是已知真值，训练和推理时模型并不知道 x0，也拿不到真实 ε；它只能根据 xt、时间步及条件估计噪声。手算验证了公式之间的一致性，不能证明一个未训练的模型能恢复图像。

#### 噪声预测误差如何传到反向均值

继续使用第二步的参数，反向均值中噪声预测的系数是 β2 除以 √α2√(1-ᾱ2)，约 0.4226。若真值噪声是 -0.5，模型给出 -0.3，预测误差为 0.2；代入反向均值后，结果相对真噪声预测会偏移约 -0.4226×0.2=-0.0845。这个小算例把“噪声预测准确度”与“生成状态误差”连接起来：不同时间步的系数不同，同样大小的预测误差可能造成不同程度的状态偏移。

但这并不意味着可以直接把平均 MSE 换算成图像质量。真实模型预测的是高维 latent 或像素张量，误差具有空间结构；同样的均方误差若集中在主体边缘、文字或背景，感知影响不同。采样器还会在多步中传播误差，且文本条件和 guidance 会改变轨迹。MSE 是训练目标或局部诊断量，图像质量仍须用任务相关评估、人工切片和多随机种子观察。

做采样器对比时，固定 checkpoint、初始随机种子、prompt 和图像预处理，再分别报告采样步数、每次模型调用、延迟、显存及质量。若不同方法使用了不同的计算预算，必须公开预算；否则更少步数的质量差异无法归因于采样规则本身。手算系数用于理解传播路径，真实结论由目标模型和目标任务的实验支撑。

在第二步，信号方差占比是 ᾱ2=0.72，噪声方差占比是 1-ᾱ2=0.28，因此 SNR=0.72/0.28≈2.57。信号仍大于噪声；继续加噪后，ᾱ 会下降，SNR 也会下降。这个比例解释了为何不同时间步的去噪难度不一样，也解释了训练时要让模型看到多个 t，而不是只拟合一个固定噪声强度。

如果评估只在靠前的低噪声时间步测 MSE，模型仍可能在接近纯噪声的高 t 区间失败。训练和诊断可以按时间步分桶比较噪声误差、预测 x0 的误差和采样结果，但这些仍是不同层次的指标。最终任务若关心文字清晰度、主体结构或风格一致性，还要把结果按这些属性单独评估。

### 68.6 生成评估与失败归因

图像生成评估不能只看 CLIP score。需要至少拆成：

~~~text
prompt adherence：是否包含要求的主体、属性和关系
图像质量：清晰度、伪影、构图和局部细节
多样性：不同随机种子是否塌缩
安全：不应生成的内容是否出现
文字与数字：渲染是否正确
成本：采样步数、显存、延迟和吞吐
~~~

FID、Inception Score 和 CLIP score 各有假设，不能互相替代，也不能直接替代人工判断。图像中数字、表格和文字尤其容易出错，若业务依赖这些字段，应使用 OCR 或结构化校验，而不是把一个整体相似度当作可靠性证明。

### 68.7 资料与证据边界

扩散基础见 [DDPM](https://arxiv.org/abs/2006.11239)，快速采样见 [DDIM](https://arxiv.org/abs/2010.02502)，latent diffusion 见 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)，DiT 见 [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)，classifier-free guidance 见 [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)。论文中的生成质量、采样步数和数据集结论不能直接转成目标业务的质量或安全保证。

本章的输出需按[第十二章多模态评估与交付](12-多模态评估与交付.md)记录来源、边界和失败；多模态机制的系统主讲见[第十五册](../../book-15-multimodal-generative-models/目录.md)。
