# 第六章：Diffusion 基础

Diffusion 是现代图像生成模型的核心路线之一。Stable Diffusion、Imagen、DALL·E 的部分路线、视频生成模型和很多 3D/音频生成方法，都与 diffusion 或相邻的去噪生成思想有关。它的直觉很简单：训练时把真实数据逐步加噪，模型学习如何去噪；生成时从随机噪声出发，反复去噪，最后得到图片、视频或音频。

本章目标是建立 diffusion 的基础图景：forward diffusion、reverse denoising、噪声预测、noise scheduler、U-Net、条件生成、classifier-free guidance、采样步数和常见面试表达。为了照顾小白，本章尽量用直觉和最少公式讲清楚，不追求完整数学推导。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，重点校准了 DDPM、DDIM、score-based generative modeling、classifier guidance / classifier-free guidance 和 latent diffusion 的代表论文资料。它们共同构成现代 diffusion 图像生成的基础路线：先人为定义加噪过程，再学习反向去噪过程；采样时用 scheduler 控制去噪路径；条件生成时用文本、类别或图像条件影响每一步去噪。

本讲只讲 diffusion 基础，不提前展开下一章 Stable Diffusion 与 DALL·E 的系统结构、prompt 工程、negative prompt、ControlNet、图像编辑和产品化参数。本讲重点放在 DDPM 的 forward / reverse 公式、噪声预测目标、timestep、scheduler、CFG、latent diffusion 成本直觉和最小可运行加噪 / 去噪审计 demo。

第二轮新增内容按三个目标补齐：

1. 把 forward diffusion、闭式加噪、噪声预测 loss、从噪声还原 `x0`、DDPM 采样均值、score 关系和 CFG 写成可复习公式。
2. 给出一个 0 依赖 Python demo，帮助读者手算 alpha schedule、`x_t`、noise MSE、`x0_hat`、DDPM reverse mean、CFG 和 latent cost ratio。
3. 同步第四册百科、题库、练习、术语表、项目路线、知识图谱和进度记录中的 diffusion 基础审计入口。

## 6.1 生成模型要解决什么问题

生成模型的目标是学习数据分布，并从中采样新样本。

例如图像生成：

```text
训练数据：大量真实图片
目标：生成一张看起来像真实图片的新图片
```

文本到图像生成进一步要求：

```text
prompt: 一只穿宇航服的猫在月球上
output: 符合描述的图片
```

早期生成模型包括 VAE、GAN、自回归图像模型等。Diffusion 的优势是训练相对稳定、生成质量高、可控性强，因此成为图像生成主流路线之一。

面试回答：

```text
Diffusion 是一种生成模型。训练时把真实样本逐步加噪，让模型学习在给定噪声程度和条件下预测噪声或去噪方向；生成时从随机噪声开始，反复调用去噪模型逐步得到清晰样本。文生图时，文本 prompt 作为条件控制生成内容。
```

## 6.2 Diffusion 的核心直觉

Diffusion 可以分成两个过程：

1. Forward diffusion：把真实图片逐步加噪，直到接近纯噪声。
2. Reverse denoising：从噪声开始逐步去噪，恢复成图片。

直觉：

```text
真实图片 -> 加一点噪声 -> 加更多噪声 -> ... -> 纯噪声
纯噪声 -> 去一点噪声 -> 更清晰 -> ... -> 生成图片
```

训练时，forward 加噪过程是人为定义的；模型真正要学的是 reverse 去噪过程。

## 6.3 Forward Diffusion：逐步加噪

假设真实图片是 `x0`。加噪后得到 `xt`，其中 `t` 表示噪声强度或时间步。

直觉上：

```text
t 小：图片还比较清楚
t 大：图片几乎全是噪声
```

训练时会随机选择一个时间步 `t`，向图片加入对应强度的噪声。

伪代码：

```python
noise = torch.randn_like(x0)
t = sample_timestep()
xt = add_noise(x0, noise, t)
```

模型输入：

```text
noisy image xt + timestep t + condition
```

模型目标：

```text
预测加入的 noise
```

### 6.3.1 关键公式与 DDPM 速查

DDPM 的 forward diffusion 通常定义为一个固定的高斯加噪马尔可夫链：

```math
q(x_t\mid x_{t-1})=
\mathcal{N}\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_t I
\right)
```

其中：

```math
\alpha_t=1-\beta_t,\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
```

`beta_t` 是第 `t` 步加多少噪声，`alpha_t` 是保留多少信号，`alpha_bar_t` 是从第 1 步到第 `t` 步累计保留下来的信号比例。

一个重要便利是：不需要真的一步一步加噪，可以直接从 `x0` 采样任意时间步 `t` 的 noisy sample：

```math
q(x_t\mid x_0)=
\mathcal{N}\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
```

等价写法是：

```math
x_t=
\sqrt{\bar{\alpha}_t}x_0+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
```

训练时最常见的简化目标是预测噪声：

```math
L_{\mathrm{simple}}=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,c)
\right\|_2^2
\right]
```

其中 `c` 是条件，可以为空、类别标签、文本 embedding 或其他控制信号。预测出噪声后，可以估计干净样本：

```math
\hat{x}_0=
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t,c)
}{
\sqrt{\bar{\alpha}_t}
}
```

DDPM 采样时常用模型预测噪声来构造反向均值：

```math
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t,c)
\right)
```

然后从反向高斯分布采样：

```math
p_\theta(x_{t-1}\mid x_t)=
\mathcal{N}\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2 I
\right)
```

score-based 视角里，噪声预测和 score 有简单关系：

```math
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
```

所以“预测噪声”“预测 score”“预测去噪方向”本质上是高度相关的训练口径，只是参数化方式不同。

## 6.4 Reverse Denoising：逐步去噪

生成时没有真实图片，只有随机噪声。

```python
x = torch.randn(batch_size, channels, height, width)
```

然后从大时间步到小时间步反复去噪：

```python
for t in reversed(timesteps):
    pred_noise = model(x, t, condition)
    x = scheduler.step(pred_noise, t, x)
```

每一步都让样本更接近数据分布。

最终得到：

```text
generated image
```

这个过程和语言模型自回归生成不同。语言模型是一个 token 一个 token 生成；diffusion 是从整体噪声图像逐步 refine。

## 6.5 模型到底预测什么

Diffusion 模型常见预测目标包括：

1. 预测噪声 `epsilon`。
2. 预测干净样本 `x0`。
3. 预测 velocity `v`。
4. 预测 score 或去噪方向。

最常见的入门理解是预测噪声。

训练目标：

```python
pred_noise = model(xt, t, condition)
loss = ((pred_noise - noise) ** 2).mean()
```

也就是说，模型看到一张被加噪的图片和时间步，要预测“这次加进去的噪声是什么”。如果能预测噪声，就能把噪声从图像中去掉。

## 6.6 Timestep 和时间嵌入

同一张噪声图在不同时间步含义不同。模型必须知道当前噪声强度。

因此 diffusion 模型会输入 timestep。

常见做法是把 `t` 编码成 embedding：

```text
timestep -> time embedding -> denoiser
```

时间步告诉模型：

1. 当前图像有多吵。
2. 应该去掉多少噪声。
3. 当前阶段是粗轮廓生成还是细节修复。

早期大时间步更像生成大结构，后期小时间步更像修细节。

## 6.7 Noise Scheduler

Noise scheduler 定义加噪和去噪的时间表。

它控制：

1. 每个时间步噪声强度。
2. 训练时如何从 `x0` 得到 `xt`。
3. 采样时如何从 `xt` 更新到更干净的 `x`。
4. 使用多少采样步。

常见 scheduler 或采样方法包括：

1. DDPM。
2. DDIM。
3. Euler。
4. DPM-Solver。
5. Heun。

面试中不一定要背每个 scheduler 的公式，但要知道：scheduler 是 diffusion 采样质量、速度和稳定性的关键控制组件。

## 6.8 U-Net Denoiser

早期图像 diffusion 常用 U-Net 作为去噪网络。

U-Net 的特点：

1. 编码器逐步下采样，提取高层语义。
2. 解码器逐步上采样，恢复空间分辨率。
3. skip connection 保留局部细节。
4. 可以注入 timestep embedding 和文本条件。

结构直觉：

```text
noisy image -> down blocks -> middle block -> up blocks -> predicted noise
```

为什么适合图像？因为去噪既需要全局语义，也需要局部像素细节。U-Net 的多尺度结构天然适合这件事。

## 6.9 DiT：Transformer Denoiser

除了 U-Net，现代路线也使用 Diffusion Transformer，简称 DiT。

思路：

1. 把图像或 latent 切成 patch tokens。
2. 用 Transformer 建模 token 间关系。
3. 预测噪声或 velocity。

DiT 的优势：

1. 更符合大规模 Transformer 扩展规律。
2. 和 LLM/Vision Transformer 架构更统一。
3. 在大规模训练下表现强。

代价：

1. 计算成本高。
2. 对训练数据和工程优化要求高。

## 6.10 条件生成

Diffusion 可以是无条件生成，也可以是条件生成。

条件可以是：

1. 文本 prompt。
2. 类别标签。
3. 图片。
4. 边缘图。
5. 深度图。
6. 姿态。
7. 音频。

文生图时，文本条件通常来自 text encoder。

```text
prompt -> text encoder -> text embeddings -> denoiser cross-attention
```

U-Net 或 DiT 在去噪时会使用这些文本 embedding，让生成结果符合 prompt。

## 6.11 Cross-Attention 如何注入文本条件

Stable Diffusion 这类模型中，文本条件常通过 cross-attention 注入 U-Net。

直觉：

1. 图像 latent 特征作为 query。
2. 文本 tokens 作为 key/value。
3. 图像在去噪时 attend 到 prompt 中的词。

例如 prompt：

```text
a red car on the street
```

模型在生成汽车区域时，会通过 cross-attention 使用 `red`、`car`、`street` 等文本信息。

这也是为什么 text encoder、prompt 细节和 attention 机制会影响生成内容。

## 6.12 Classifier-Free Guidance

Classifier-Free Guidance，简称 CFG，是文生图中常见的控制技术。

训练时，模型有时接收文本条件，有时把条件置空。这样模型学会：

1. 有条件去噪。
2. 无条件去噪。

生成时，同时计算：

1. 条件预测。
2. 无条件预测。

然后增强条件方向。

直觉：

```text
guided_prediction = unconditional + scale * (conditional - unconditional)
```

`scale` 越大，越强调 prompt 条件；但太大可能导致图像过饱和、失真或细节异常。

公式写法如下：

```math
\hat{\epsilon}_{\mathrm{cfg}}=
\epsilon_\theta(x_t,t,\varnothing)
w\left(
\epsilon_\theta(x_t,t,c)
-
\epsilon_\theta(x_t,t,\varnothing)
\right)
```

其中 `w` 是 guidance scale。`w=1` 时通常接近条件预测；`w` 继续增大时，会沿着“条件预测减无条件预测”的方向更用力地走。

面试回答：

```text
Classifier-free guidance 通过比较有条件和无条件去噪预测，增强朝向文本条件的方向。guidance scale 越大，生成越贴近 prompt，但过大可能带来过饱和、失真和多样性下降。
```

## 6.13 Latent Diffusion

直接在像素空间做 diffusion 成本很高。Latent Diffusion 的思路是先用 autoencoder 把图片压缩到 latent space，再在 latent 中扩散和去噪。

流程：

```text
image -> VAE encoder -> latent
latent diffusion denoising
latent -> VAE decoder -> image
```

优点：

1. latent 分辨率更小。
2. 训练和推理更省计算。
3. 适合高分辨率图像生成。

Stable Diffusion 就是 latent diffusion 的代表。

粗略成本直觉是：如果 VAE 把空间分辨率压缩 `f` 倍，那么 spatial token / cell 数大约下降到原来的：

```math
R_{\mathrm{latent}}\approx\frac{1}{f^2}
```

实际还要考虑 latent channel 数、U-Net / DiT 宽度和采样步数，但这个式子足以解释为什么在 latent space 做 diffusion 更适合高分辨率图像生成。

## 6.14 采样步数和质量速度 trade-off

Diffusion 生成通常需要多步采样。

步数越多：

1. 通常质量更稳定。
2. 生成更慢。

步数越少：

1. 速度更快。
2. 质量可能下降。

现代 scheduler 和蒸馏方法尝试用更少步数达到较好质量。例如从几十步减少到几步甚至一步，但通常要额外训练或质量折中。

## 6.15 Diffusion 和自回归生成的区别

自回归语言模型：

```text
token_1 -> token_2 -> token_3 -> ...
```

Diffusion：

```text
noise image -> denoise step -> denoise step -> final image
```

区别：

1. 自回归按序生成离散 token。
2. Diffusion 从整体噪声逐步 refine。
3. 自回归天然适合文本序列。
4. Diffusion 天然适合连续图像和音频。
5. Diffusion 采样步数通常较多。

## 6.16 一个最小训练伪代码

```python
for images, captions in dataloader:
    noise = torch.randn_like(images)
    timesteps = sample_timesteps(images.size(0))
    noisy_images = scheduler.add_noise(images, noise, timesteps)

    text_embeddings = text_encoder(captions)
    pred_noise = denoiser(noisy_images, timesteps, text_embeddings)

    loss = ((pred_noise - noise) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

这段伪代码说明 diffusion 训练最核心的闭环：采样噪声和时间步、加噪、预测噪声、用 MSE 训练。

## 6.17 一个最小采样伪代码

```python
x = torch.randn(batch_size, channels, height, width)
text_embeddings = text_encoder(prompts)

for t in scheduler.timesteps:
    pred_noise = denoiser(x, t, text_embeddings)
    x = scheduler.step(pred_noise, t, x)

images = x
```

如果是 latent diffusion，最后还要通过 VAE decoder：

```python
images = vae.decode(latents)
```

## 6.18 常见失败模式

Diffusion 生成常见问题：

1. 手指、文字等细节错误。
2. prompt 不遵循。
3. 多物体关系混乱。
4. 文字生成乱码。
5. 人脸或身体结构异常。
6. 风格过拟合。
7. guidance scale 过高导致失真。
8. 采样步数太少导致质量差。

这些问题来自训练数据、模型容量、条件注入、采样策略和生成任务本身的复杂性。

## 6.19 面试官会怎么问

### 问题一：Diffusion 的核心思想是什么？

回答模板：

```text
Diffusion 的核心是加噪和去噪。训练时把真实样本按不同时间步加噪，模型学习预测噪声或去噪方向；生成时从随机噪声开始，按 scheduler 多步去噪，逐渐得到清晰样本。文生图中，文本 prompt 作为条件控制去噪过程。
```

### 问题二：为什么模型常常预测噪声？

回答模板：

```text
训练时我们知道加到真实图片上的噪声，所以可以构造监督信号。模型输入 noisy image 和 timestep，预测加入的噪声，用 MSE 训练。预测出噪声后，就可以在采样时逐步去掉噪声，恢复样本。
```

### 问题三：Noise scheduler 的作用是什么？

回答模板：

```text
Scheduler 定义每个时间步的噪声强度，以及采样时如何从当前 noisy sample 更新到下一步。它影响训练加噪过程、推理采样路径、生成质量和速度，是 diffusion 系统中的关键组件。
```

### 问题四：Classifier-free guidance 是什么？

回答模板：

```text
CFG 让模型同时学习有条件和无条件去噪。生成时比较条件预测和无条件预测，并放大朝向条件的方向，使结果更符合 prompt。guidance scale 太低可能不跟 prompt，太高可能失真和多样性下降。
```

### 问题五：Latent Diffusion 为什么更高效？

回答模板：

```text
因为它不直接在高维像素空间扩散，而是先用 VAE 把图片压缩到低维 latent，在 latent space 中去噪，最后再解码回图片。这样显著降低计算和显存成本，适合高分辨率生成。
```

## 6.20 小练习

1. 用自己的话解释 forward diffusion 和 reverse denoising。
2. 写一个伪代码，随机采样 timestep 并给图片加噪。
3. 写出预测噪声的 MSE loss。
4. 解释 timestep embedding 为什么必要。
5. 比较 U-Net 和 DiT 作为 denoiser 的区别。
6. 解释 CFG scale 太大可能带来的问题。
7. 画出 latent diffusion 的训练和推理流程。
8. 比较 diffusion 和自回归语言模型生成方式的差异。
9. 写一个 0 依赖 Python demo，手算 alpha schedule、闭式加噪、噪声预测 loss、`x0_hat`、CFG 和 latent cost ratio。

## 6.21 最小可运行 diffusion 加噪 / 去噪审计 demo

下面这个 demo 不依赖深度学习框架，用 4 维 toy 向量模拟 DDPM 的几个核心检查：`alpha_bar` 是否递减、闭式加噪是否保持 shape、噪声预测 MSE、从预测噪声恢复 `x0`、DDPM reverse mean、score 关系、CFG 和 latent diffusion 成本比例。

```python
import math


def round_list(values, ndigits=4):
    return [round(value, ndigits) for value in values]


x0 = [0.2, -0.4, 0.7, -0.1]
noise = [0.5, -1.0, 0.25, 0.75]
pred_noise = [0.45, -0.9, 0.3, 0.7]
betas = [0.0001, 0.005, 0.01, 0.015, 0.02]

alphas = [1.0 - beta for beta in betas]
alpha_bars = []
running = 1.0
for alpha in alphas:
    running *= alpha
    alpha_bars.append(running)

t = 4
index = t - 1
alpha_bar_t = alpha_bars[index]
sqrt_alpha_bar = math.sqrt(alpha_bar_t)
sqrt_one_minus_alpha_bar = math.sqrt(1.0 - alpha_bar_t)

x_t = [
    sqrt_alpha_bar * clean + sqrt_one_minus_alpha_bar * eps
    for clean, eps in zip(x0, noise)
]

noise_mse = sum(
    (pred - eps) ** 2 for pred, eps in zip(pred_noise, noise)
) / len(noise)

x0_hat = [
    (noisy - sqrt_one_minus_alpha_bar * pred) / sqrt_alpha_bar
    for noisy, pred in zip(x_t, pred_noise)
]
reconstruction_mae = sum(
    abs(recon - clean) for recon, clean in zip(x0_hat, x0)
) / len(x0)

beta_t = betas[index]
alpha_t = alphas[index]
ddpm_mean = [
    (noisy - beta_t / sqrt_one_minus_alpha_bar * pred) / math.sqrt(alpha_t)
    for noisy, pred in zip(x_t, pred_noise)
]

score = [-eps / sqrt_one_minus_alpha_bar for eps in noise]

uncond_eps = [0.7, -0.8, 0.2, 0.5]
cond_eps = [0.4, -1.1, 0.3, 0.8]
guidance_scale = 3.0
cfg_guided_eps = [
    uncond + guidance_scale * (cond - uncond)
    for uncond, cond in zip(uncond_eps, cond_eps)
]

pixel_values = 512 * 512 * 3
latent_values = (512 // 8) * (512 // 8) * 4
latent_ratio = latent_values / pixel_values

checks = {
    "alpha_bar_decreases": all(
        alpha_bars[i] > alpha_bars[i + 1] for i in range(len(alpha_bars) - 1)
    ),
    "xt_shape_ok": len(x_t) == len(x0),
    "mse_positive": noise_mse > 0,
    "reconstruction_close": reconstruction_mae < 0.02,
    "cfg_changes_prediction": cfg_guided_eps != cond_eps
    and cfg_guided_eps != uncond_eps,
    "latent_is_smaller": latent_ratio < 0.05,
}

print("alpha_bars=", round_list(alpha_bars, 6))
print("t=", t, "alpha_bar_t=", round(alpha_bar_t, 6))
print("x_t=", round_list(x_t, 4))
print("noise_mse=", round(noise_mse, 5))
print("x0_hat=", round_list(x0_hat, 4))
print("reconstruction_mae=", round(reconstruction_mae, 5))
print("ddpm_mean=", round_list(ddpm_mean, 4))
print("score=", round_list(score, 4))
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
print("gate_pass=", all(checks.values()))
```

一组期望输出如下：

```text
alpha_bars= [0.9999, 0.9949, 0.984951, 0.970177, 0.950774]
t= 4 alpha_bar_t= 0.970177
x_t= [0.2833, -0.5667, 0.7327, 0.031]
noise_mse= 0.00437
x0_hat= [0.2088, -0.4175, 0.6912, -0.0912]
reconstruction_mae= 0.01096
ddpm_mean= [0.2461, -0.4922, 0.712, -0.03]
score= [-2.8953, 5.7906, -1.4477, -4.343]
cfg_guided_eps= [-0.2, -1.7, 0.5, 1.4]
latent_values= 16384 pixel_values= 786432 latent_ratio= 0.0208
sampling_calls= {'ddpm_50': 50, 'ddim_20': 20, 'distilled_4': 4}
checks= {'alpha_bar_decreases': True, 'xt_shape_ok': True, 'mse_positive': True, 'reconstruction_close': True, 'cfg_changes_prediction': True, 'latent_is_smaller': True}
gate_pass= True
```

面试表达：

```text
我会把 diffusion 基础拆成几个可验证量：beta schedule 产生 alpha_bar，alpha_bar 决定任意时间步的闭式加噪；模型预测噪声，用 MSE 训练；预测噪声可以还原 x0，也可以构造 DDPM 反向均值；CFG 是把条件预测和无条件预测做方向增强；latent diffusion 则用压缩空间降低每一步去噪成本。
```

## 6.22 本章总结

Diffusion 的核心主线是：训练时给真实样本加噪，模型学习去噪；生成时从随机噪声出发，多步去噪得到样本。文生图中，文本条件通过 text encoder 和 cross-attention 等方式注入 denoiser，CFG 用来增强 prompt 遵循，latent diffusion 用低维 latent 降低生成成本。

需要记住：

1. Forward diffusion 是人为定义的加噪过程。
2. Reverse denoising 是模型要学习的生成过程。
3. 入门理解中，模型常预测噪声。
4. Timestep 告诉模型当前噪声强度。
5. Scheduler 控制加噪和采样路径。
6. U-Net 和 DiT 都可以作为 denoiser。
7. CFG 提升 prompt 遵循，但过强会损害质量。
8. Latent diffusion 在压缩后的 latent 空间生成，更高效。
9. DDPM 基础面试至少要能写出闭式加噪、噪声预测 loss 和 CFG 公式。

下一章会进入 Stable Diffusion 与 DALL·E，讲清主流文生图系统的结构、VAE、text encoder、U-Net/DiT、prompt、negative prompt、ControlNet 和图像编辑。
