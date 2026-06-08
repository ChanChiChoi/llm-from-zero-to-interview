# 第七章：Stable Diffusion 与 DALL·E

上一章讲了 diffusion 的基础：加噪、去噪、scheduler、U-Net、条件生成和 classifier-free guidance。本章进入主流文生图系统，重点讲 Stable Diffusion 和 DALL·E。它们代表了两条重要路线：Stable Diffusion 以 latent diffusion 为核心，通过 VAE、text encoder、U-Net/DiT 和 scheduler 生成图像；DALL·E 系列则经历了从离散图像 token 自回归生成到更强的文生图系统演进。

本章目标不是复现完整模型，而是讲清一个文生图系统由哪些模块组成、prompt 如何影响生成、negative prompt 和 CFG 如何工作、ControlNet 和图像编辑为什么重要，以及自回归图像生成和 diffusion 图像生成的差异。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，重点核对了 Latent Diffusion / Stable Diffusion 论文、DALL·E 与 DALL·E 2 论文、classifier-free guidance、ControlNet、DiT / Stable Diffusion 3 相关论文，以及 Hugging Face Diffusers 中 Stable Diffusion pipeline 的常见参数口径。正文只吸收适合面试和工程理解的稳定结论，不追具体闭源产品的未公开训练细节。

本章聚焦：

1. Stable Diffusion 类系统的 VAE、text encoder、denoiser、scheduler 和 VAE decoder 分工。
2. latent space 为什么降低成本，以及它和像素空间 diffusion 的差异。
3. prompt、negative prompt、CFG、scheduler、seed、image-to-image、inpainting 和 ControlNet 的机制边界。
4. DALL·E 早期自回归离散图像 token 路线、DALL·E 2 的 CLIP latent 两阶段路线，以及与 diffusion 路线的对比。
5. 可手算的文生图 pipeline 审计：latent shape、VAE 压缩比、U-Net 调用次数、cross-attention 成本、ControlNet 条件和自回归图像 token 成本。

本章不展开完整 U-Net 结构、VAE 训练细节、DiT / rectified flow 全量推导、生产级安全系统、商业模型内部策略、图像版权法务细节或高质量图片真实生成代码。这里的 Python demo 是审计 pipeline 形状、成本和参数关系的教学脚本，不是生成图片的模型实现。

## 7.1 文生图系统解决什么问题

文生图的输入是文本 prompt，输出是图片。

例如：

```text
prompt: a watercolor painting of a robot reading a book in a garden
output: 一张符合描述的图片
```

模型需要同时处理：

1. 文本理解。
2. 视觉语义生成。
3. 空间布局。
4. 风格控制。
5. 细节一致性。
6. 安全过滤。

文生图系统通常不是一个单独网络，而是一组模块协同工作。

面试回答：

```text
文生图系统通常由 text encoder、生成主干、scheduler 和图像解码模块组成。以 Stable Diffusion 为例，文本 prompt 先被 text encoder 编码，U-Net 在 latent space 中根据文本条件逐步去噪，scheduler 控制采样路径，最后 VAE decoder 把 latent 解码成图片。
```

## 7.2 Stable Diffusion 的总体结构

Stable Diffusion 是 latent diffusion 的代表。

核心模块：

1. VAE encoder。
2. VAE decoder。
3. Text encoder。
4. Denoising U-Net 或 DiT。
5. Noise scheduler。

训练直觉：

```text
image -> VAE encoder -> latent
latent + noise + timestep + text condition -> U-Net predicts noise
```

推理直觉：

```text
prompt -> text encoder -> text embeddings
random latent noise -> U-Net denoising steps -> clean latent -> VAE decoder -> image
```

为什么叫 latent diffusion？因为扩散过程不直接发生在像素空间，而是在 VAE 压缩后的 latent space 中进行。

### 7.2.1 关键公式与文生图 pipeline 速查

Stable Diffusion 类系统可以先记住一个总流程：

```text
prompt -> tokenizer -> text encoder -> condition
noise latent -> denoiser + scheduler 多步去噪 -> clean latent
clean latent -> VAE decoder -> image
```

**VAE 压缩与解码**

设输入图像为 `x`，VAE encoder 为 `E_phi`，decoder 为 `D_psi`，latent 空间压缩倍数为 `f`。如果图像尺寸是 `H x W`，latent 的空间尺寸通常约为 `H/f x W/f`。

```math
z=E_\phi(x),
\qquad
\hat{x}=D_\psi(z)
```

```math
z\in\mathbb{R}^{B\times C_z\times H/f\times W/f}
```

只看空间 cell，latent diffusion 相比像素空间的空间成本比例约为：

```math
R_{\mathrm{spatial}}=
\frac{(H/f)(W/f)}{HW}
=
\frac{1}{f^2}
```

如果把通道数也算进去，元素级压缩比可以写成：

```math
R_{\mathrm{elem}}=
\frac{C_z(H/f)(W/f)}{3HW}
```

其中 3 表示 RGB 三通道。面试时要说明：`R_spatial` 解释 diffusion 主干的空间网格变小，`R_elem` 才是把 latent channels 也算进去的粗略元素量对比。

**文本条件与去噪目标**

prompt 经过 text encoder 得到条件表示 `E_c`：

```math
E_c=T_\eta(p)
```

denoiser 在 latent space 中预测噪声：

```math
\hat{\epsilon}=
\epsilon_\theta(z_t,t,E_c)
```

训练时常见噪声预测目标：

```math
L_{\mathrm{t2i}}=
\mathbb{E}_{z_0,t,\epsilon,p}
\left[
\left\|
\epsilon-
\epsilon_\theta(z_t,t,T_\eta(p))
\right\|_2^2
\right]
```

其中 `z_0` 是干净 latent，`z_t` 是加噪后的 latent，`p` 是 prompt。

**Cross-Attention 条件注入**

U-Net 或 Transformer denoiser 需要把图像 latent 特征和文本条件对齐。常见做法是让图像 token 做 query，文本 token 做 key/value：

```math
Q=H W_Q,\qquad K=E_c W_K,\qquad V=E_c W_V
```

```math
\mathrm{Attn}(H,E_c)=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
```

如果 latent token 数为 `N_z`，文本 token 数为 `N_c`，一次 cross-attention 的注意力矩阵规模约为：

```math
C_{\mathrm{cross}}\approx N_zN_c
```

因此长 prompt、高分辨率 latent、多步采样和多层 cross-attention 会共同推高推理成本。

**CFG 与 negative prompt**

文生图实现里，negative prompt 往往不是硬约束，而是作为负条件 `c_neg` 参与 CFG。设正向 prompt 条件为 `c_pos`，负向或空条件为 `c_neg`：

```math
\epsilon_{\mathrm{guided}}=
\epsilon_\theta(z_t,t,c_{\mathrm{neg}})
s
\left(
\epsilon_\theta(z_t,t,c_{\mathrm{pos}})
-
\epsilon_\theta(z_t,t,c_{\mathrm{neg}})
\right)
```

其中 `s` 是 guidance scale。`s=1` 时近似不额外放大条件方向；`s` 过大时可能更贴 prompt，但也更容易过饱和、失真和多样性下降。

**Image-to-Image 强度**

Image-to-image 会先把输入图像编码到 latent，再加到某个噪声强度，然后从中间时间步开始去噪：

```math
z_t=
\sqrt{\bar{\alpha}_t}z_0+
\sqrt{1-\bar{\alpha}_t}\epsilon
```

denoising strength 越大，起始时间步越靠近高噪声区，模型越自由；strength 越小，原图结构保留越多。

**ControlNet 条件残差**

ControlNet 的核心不是替换原模型，而是在冻结或复用的 diffusion backbone 旁边学习一条条件分支，把边缘、深度、姿态等结构条件变成残差注入主干：

```math
h_l^{\mathrm{out}}=
h_l^{\mathrm{base}}+
r_l(C)
```

其中 `C` 是结构条件，`r_l(C)` 是第 `l` 层的条件残差。直觉上：prompt 控制“画什么”，ControlNet 条件控制“按什么结构画”。

**DALL·E 早期自回归图像 token 目标**

早期 DALL·E 把图像离散化为 image tokens，然后和文本 tokens 一起交给 decoder-only Transformer。若文本为 `u`，图像 token 为 `y_1,...,y_M`：

```math
p(y_1,\ldots,y_M\mid u)=
\prod_{m=1}^{M}
p(y_m\mid u,y_{1:m-1})
```

训练目标是 next-token negative log likelihood：

```math
L_{\mathrm{ar}}=
-
\sum_{m=1}^{M}
\log p_\theta(y_m\mid u,y_{1:m-1})
```

这条路线的优点是和语言模型形式统一，缺点是高分辨率图像 token 序列长，生成速度和 tokenizer 质量压力都很大。

**DALL·E 2 的 CLIP latent 两阶段直觉**

DALL·E 2 更像 `text -> CLIP image latent -> image decoder`：

```math
z_{\mathrm{clip}} \sim p_\theta(z_{\mathrm{clip}}\mid p),
\qquad
x \sim p_\psi(x\mid z_{\mathrm{clip}},p)
```

其中 prior 负责从文本条件生成 CLIP image latent，decoder 负责把该图像语义 latent 生成图片。它和 Stable Diffusion 的区别在于条件组织方式不同：Stable Diffusion 类系统通常直接用 text encoder 表示指导 latent denoising，而 DALL·E 2 论文强调先生成 CLIP image representation，再由 decoder 生成图像。

## 7.3 VAE 的作用

VAE 在 Stable Diffusion 中负责像素空间和 latent space 的转换。

```text
image -> VAE encoder -> latent
latent -> VAE decoder -> image
```

优点：

1. latent 分辨率更小。
2. 扩散模型计算更省。
3. 支持较高分辨率生成。

例如一张 `512 x 512` 图片，经过 VAE encoder 后可能变成更小的 latent 表示。U-Net 在 latent 上去噪，最后再由 VAE decoder 还原成图片。

VAE 的质量会影响最终图片：

1. 解码器太弱会导致细节损失。
2. 压缩太强会影响文字、纹理和边缘。
3. latent 空间设计会影响生成效率和质量。

## 7.4 Text Encoder 的作用

Text encoder 把 prompt 编码成条件向量。

```text
prompt -> tokenizer -> text encoder -> text embeddings
```

Stable Diffusion 常见路线使用 CLIP text encoder，也有系统使用更强的 T5 类文本编码器。

Text encoder 影响：

1. prompt 理解能力。
2. 长 prompt 处理能力。
3. 组合关系表达。
4. 多语言能力。
5. 风格和细节控制。

如果 text encoder 对 prompt 理解不足，后面的图像生成模块再强也可能无法正确遵循指令。

## 7.5 U-Net Denoiser

Stable Diffusion 早期主干是 U-Net。

输入：

```text
noisy latent + timestep + text condition
```

输出：

```text
predicted noise
```

U-Net 中通常通过 cross-attention 注入文本条件。图像 latent 特征作为 query，文本 embedding 作为 key/value，让去噪过程根据 prompt 调整内容。

U-Net 的多尺度结构适合图像生成：

1. 低分辨率层负责全局布局。
2. 高分辨率层负责局部细节。
3. skip connection 保留空间信息。

## 7.6 DiT 和新一代文生图主干

随着 Transformer 扩展规律在视觉生成中变强，越来越多模型使用 DiT 或类似 Transformer denoiser。

DiT 思路：

1. 把 latent 切成 patch tokens。
2. 用 Transformer 处理。
3. 根据 timestep 和文本条件预测噪声或 velocity。

优点：

1. 扩展性好。
2. 架构统一。
3. 更容易利用大规模训练经验。

代价：

1. 训练成本高。
2. 推理优化复杂。
3. 对数据规模和工程系统要求高。

面试中可以说：U-Net 是 Stable Diffusion 早期核心，DiT 是更现代的大规模 diffusion transformer 路线。

## 7.7 Prompt 和 Negative Prompt

Prompt 描述想生成什么。

```text
a high quality photo of a golden retriever, natural light, detailed fur
```

Negative prompt 描述不希望出现什么。

```text
blurry, low quality, distorted face, extra fingers
```

从机制上看，negative prompt 通常会作为“负条件”参与 CFG 方向计算，帮助模型远离某些不希望出现的特征。

注意：negative prompt 不是魔法。它可以改善常见质量问题，但不能保证严格约束，也不能解决所有结构错误。

## 7.8 Classifier-Free Guidance 在文生图中的作用

CFG 是 Stable Diffusion 中非常核心的推理技巧。

生成时模型会计算：

1. 无条件预测。
2. 有条件预测。

然后放大条件方向：

```text
guided = uncond + scale * (cond - uncond)
```

`scale` 较低：

1. 多样性更强。
2. prompt 遵循可能较弱。

`scale` 较高：

1. 更贴近 prompt。
2. 可能过饱和、失真、细节崩坏。

工程上需要根据模型和任务调整 guidance scale。

## 7.9 Scheduler 和采样器

Scheduler 决定从噪声到图像的采样路径。

常见采样器：

1. DDIM。
2. Euler。
3. Heun。
4. DPM-Solver。
5. UniPC。

影响：

1. 采样速度。
2. 图像质量。
3. 细节稳定性。
4. prompt 遵循。
5. 随机性和多样性。

同一个 prompt、seed 和模型，换 scheduler 可能得到不同风格和质量的结果。

## 7.10 Seed 和可复现性

文生图常用 seed 控制随机噪声初始状态。

相同条件下：

```text
same prompt + same seed + same model + same scheduler -> 通常生成相同或接近结果
```

但要注意：不同框架、硬件、精度、scheduler 实现可能导致细微差异。

Seed 的作用：

1. 复现实验结果。
2. 在同一构图上调整 prompt。
3. 做 A/B 对比。

## 7.11 Image-to-Image

Image-to-image 不是从纯噪声开始，而是从输入图片的 latent 加噪后再去噪。

流程：

```text
input image -> VAE encode -> latent -> add noise -> denoise with prompt -> output image
```

关键参数是 denoising strength。

强度低：

1. 更保留原图。
2. 改动较小。

强度高：

1. 改动更大。
2. 更可能偏离原图。

## 7.12 Inpainting 和 Outpainting

Inpainting 用于局部重绘。

输入：

1. 原图。
2. mask。
3. prompt。

模型只重绘 mask 区域或主要改变 mask 区域。

Outpainting 用于向外扩展图片边界。

这类任务要求模型同时保持原图一致性和生成新区域。

常见难点：

1. 边界融合。
2. 光照一致。
3. 结构连续。
4. 语义不冲突。

## 7.13 ControlNet

ControlNet 用于给 diffusion 模型加入更强的结构控制。

条件可以是：

1. 边缘图。
2. 深度图。
3. 姿态骨架。
4. 分割图。
5. 草图。

直觉：

```text
prompt 控制语义
ControlNet 条件控制结构
```

例如输入一张姿态骨架图，模型生成符合这个姿态的人物图片。

ControlNet 的价值在于：仅靠 prompt 很难精确控制布局和结构，而额外条件可以显著提升可控性。

## 7.14 LoRA 和个性化生成

图像生成中常用 LoRA 做低成本风格或角色适配。

用途：

1. 特定人物。
2. 特定画风。
3. 特定产品。
4. 特定场景。

LoRA 的优点：

1. 参数少。
2. 训练成本低。
3. 可以和 base model 组合。

风险：

1. 过拟合。
2. 泛化差。
3. 风格污染。
4. 版权和身份安全问题。

## 7.15 DALL·E 路线概览

DALL·E 系列代表了 OpenAI 在文本到图像生成上的重要路线。

早期 DALL·E 更接近自回归图像 token 生成：

```text
text tokens + image tokens -> autoregressive Transformer
```

它需要先把图像离散化成 image tokens，然后像语言模型一样预测 token 序列。

后续 DALL·E 系列融合了更强的图文理解、扩散生成、编辑和安全机制。

面试中不一定需要背每代细节，但要能比较两类思路：

1. 自回归图像 token 生成。
2. Diffusion 去噪生成。

## 7.16 自回归图像生成

自回归图像生成先把图像离散化为 token。

流程：

```text
image -> tokenizer / VQ-VAE -> image tokens
text + previous image tokens -> next image token
```

优点：

1. 和语言模型形式统一。
2. 可以直接用 next-token prediction。
3. 适合统一多模态 token 建模。

缺点：

1. 图像 token 序列很长。
2. 高分辨率成本高。
3. 生成速度可能慢。
4. 离散 tokenizer 质量很关键。

## 7.17 Diffusion 和自回归图像生成对比

Diffusion：

1. 从噪声逐步去噪。
2. 擅长连续图像生成。
3. 采样通常需要多步。
4. 可控编辑生态成熟。

自回归图像生成：

1. 按 token 顺序生成图像 token。
2. 和 LLM 训练目标统一。
3. 序列长，成本高。
4. 依赖图像 tokenizer。

没有绝对优劣，取决于目标：高质量图像生成、统一多模态建模、推理速度、编辑能力和系统复杂度。

## 7.18 文生图评估

文生图评估很难，因为质量有主观性。

常见维度：

1. 图像质量。
2. 文本一致性。
3. 美学质量。
4. 多样性。
5. 物体关系。
6. 文字生成能力。
7. 安全性。
8. 人类偏好。

常见问题：

1. FID 不一定反映 prompt 遵循。
2. CLIP score 可能被模型投机。
3. 人工评测成本高但更可靠。
4. 复杂 prompt 需要专门评估集。

## 7.19 安全和版权问题

文生图系统需要关注：

1. 色情、暴力和违法内容。
2. 名人肖像和 deepfake。
3. 版权风格模仿。
4. 商标和品牌滥用。
5. 医疗、政治等高风险误导。
6. 训练数据合规。

安全策略包括：

1. prompt filter。
2. 输出图像审核。
3. 水印和 provenance。
4. 拒绝高风险请求。
5. 数据治理。

## 7.20 最小可运行文生图 pipeline 审计 demo

下面这个 demo 不生成图片，只做文生图 pipeline 的可手算审计：输入 prompt、negative prompt、分辨率、VAE 压缩倍数、采样步数和 CFG scale，输出 latent shape、压缩比、U-Net 调用次数、cross-attention 规模、image-to-image 起始步、ControlNet 条件 shape 和 DALL·E 风格自回归图像 token 成本。

```python
from math import ceil


def simple_tokenize(text):
    return [tok for tok in text.replace(",", " ").split() if tok]


def latent_shape(batch, height, width, vae_scale, latent_channels):
    return (
        batch,
        latent_channels,
        ceil(height / vae_scale),
        ceil(width / vae_scale),
    )


def elem_count(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def audit_text_to_image_pipeline(config):
    prompt_tokens = simple_tokenize(config["prompt"])
    negative_tokens = simple_tokenize(config["negative_prompt"])
    text_tokens = min(
        config["max_text_tokens"],
        len(prompt_tokens) + config["special_tokens"],
    )
    negative_text_tokens = min(
        config["max_text_tokens"],
        len(negative_tokens) + config["special_tokens"],
    )

    z_shape = latent_shape(
        config["batch"],
        config["height"],
        config["width"],
        config["vae_scale"],
        config["latent_channels"],
    )
    latent_spatial_cells = z_shape[2] * z_shape[3]
    pixel_elements = config["batch"] * config["height"] * config["width"] * 3
    latent_elements = elem_count(z_shape)

    cfg_enabled = config["guidance_scale"] > 1.0
    unet_calls = config["steps"] * (2 if cfg_enabled else 1)
    cross_attention_cells = (
        config["steps"] * latent_spatial_cells * text_tokens
    )

    img2img_start_step = int(round(config["steps"] * config["denoising_strength"]))
    control_shape = (
        config["batch"],
        config["control_channels"],
        config["height"],
        config["width"],
    )

    dalle_image_tokens = config["dalle_grid"] ** 2
    dalle_total_tokens = (
        config["dalle_text_tokens"]
        + dalle_image_tokens
    )

    report = {
        "prompt_token_count": len(prompt_tokens),
        "negative_token_count": len(negative_tokens),
        "text_condition_shape": (
            config["batch"],
            text_tokens,
            config["text_hidden"],
        ),
        "negative_condition_shape": (
            config["batch"],
            negative_text_tokens,
            config["text_hidden"],
        ),
        "latent_shape": z_shape,
        "latent_spatial_cells": latent_spatial_cells,
        "spatial_ratio": round(latent_spatial_cells / (config["height"] * config["width"]), 6),
        "element_ratio": round(latent_elements / pixel_elements, 6),
        "unet_calls": unet_calls,
        "cross_attention_cells": cross_attention_cells,
        "img2img_start_step": img2img_start_step,
        "control_shape": control_shape,
        "dalle_total_tokens": dalle_total_tokens,
        "dalle_image_token_ratio": round(
            dalle_image_tokens / config["dalle_text_tokens"],
            3,
        ),
    }

    checks = {
        "latent_height_matches": z_shape[2] * config["vae_scale"] >= config["height"],
        "latent_width_matches": z_shape[3] * config["vae_scale"] >= config["width"],
        "cfg_doubles_unet_calls": unet_calls == config["steps"] * 2,
        "latent_ratio_small": report["element_ratio"] < 0.05,
        "control_batch_matches": control_shape[0] == config["batch"],
        "dalle_image_tokens_1024": dalle_image_tokens == 1024,
        "img2img_valid_start": 0 <= img2img_start_step <= config["steps"],
    }
    report["checks"] = checks
    report["gate_pass"] = all(checks.values())
    return report


config = {
    "prompt": "a cinematic photo of a robot reading a book in a garden, natural light",
    "negative_prompt": "blurry low quality distorted hands watermark",
    "batch": 1,
    "height": 512,
    "width": 512,
    "vae_scale": 8,
    "latent_channels": 4,
    "steps": 30,
    "guidance_scale": 7.5,
    "denoising_strength": 0.55,
    "control_channels": 1,
    "special_tokens": 2,
    "max_text_tokens": 77,
    "text_hidden": 768,
    "dalle_grid": 32,
    "dalle_text_tokens": 16,
}

report = audit_text_to_image_pipeline(config)
for key, value in report.items():
    print(f"{key}={value}")
```

一组可核对输出如下：

```text
prompt_token_count=14
negative_token_count=6
text_condition_shape=(1, 16, 768)
negative_condition_shape=(1, 8, 768)
latent_shape=(1, 4, 64, 64)
latent_spatial_cells=4096
spatial_ratio=0.015625
element_ratio=0.020833
unet_calls=60
cross_attention_cells=1966080
img2img_start_step=16
control_shape=(1, 1, 512, 512)
dalle_total_tokens=1040
dalle_image_token_ratio=64.0
checks={'latent_height_matches': True, 'latent_width_matches': True, 'cfg_doubles_unet_calls': True, 'latent_ratio_small': True, 'control_batch_matches': True, 'dalle_image_tokens_1024': True, 'img2img_valid_start': True}
gate_pass=True
```

这个 demo 对面试很有用，因为它把几个容易空谈的点变成了数字：

1. `512 x 512` 图片经 `8x` VAE 压缩后，latent 空间是 `4 x 64 x 64`。
2. CFG 会让每个采样步通常需要正负两次 denoiser 预测，因此 `30` 步对应 `60` 次 U-Net 调用。
3. cross-attention 成本同时受 latent token、文本 token 和采样步数影响。
4. ControlNet 的结构条件仍要和图像空间对齐，条件图 shape 需要被严格检查。
5. DALL·E 早期 `32 x 32` image token 网格就是 `1024` 个图像 token，和语言模型式自回归目标统一，但序列成本很高。

## 7.21 面试官会怎么问

### 问题一：Stable Diffusion 的核心模块有哪些？

回答模板：

```text
Stable Diffusion 主要包括 VAE、text encoder、denoising U-Net 或 DiT、noise scheduler。VAE 把图片压缩到 latent space 并负责解码，text encoder 编码 prompt，U-Net 在 latent space 根据文本条件预测噪声，scheduler 控制采样过程。
```

### 问题二：为什么 Stable Diffusion 在 latent space 里扩散？

回答模板：

```text
因为像素空间维度很高，直接扩散计算成本大。Latent diffusion 先用 VAE 把图片压缩到更小的 latent space，在 latent 中去噪，最后再解码成图片，这样能显著降低训练和推理成本。
```

### 问题三：Negative prompt 有什么作用？

回答模板：

```text
Negative prompt 用来描述不希望出现的内容，通常会作为负条件参与 classifier-free guidance 的方向计算，帮助模型远离低质量、模糊、畸形或不需要的特征。但它不是硬约束，不能保证完全消除问题。
```

### 问题四：ControlNet 解决什么问题？

回答模板：

```text
ControlNet 给 diffusion 模型加入额外结构条件，例如边缘、深度、姿态、分割图。Prompt 主要控制语义，ControlNet 可以更精确控制布局、姿态和结构，解决纯文本 prompt 难以精确控制生成的问题。
```

### 问题五：Diffusion 和自回归图像生成有什么区别？

回答模板：

```text
Diffusion 从噪声开始多步去噪生成图像，适合连续图像和编辑控制；自回归图像生成先把图像离散化成 tokens，再像语言模型一样逐 token 生成。自回归目标统一但序列长，diffusion 质量和编辑生态成熟但采样通常需要多步。
```

## 7.22 小练习

1. 画出 Stable Diffusion 的训练和推理流程。
2. 解释 VAE encoder 和 decoder 分别做什么。
3. 解释 text encoder 如何影响 prompt 遵循。
4. 比较 CFG scale 过低和过高的效果。
5. 设计一个 image-to-image 任务，并说明 denoising strength 的作用。
6. 说明 ControlNet 和普通 prompt 控制的区别。
7. 比较 diffusion 和自回归图像生成。
8. 设计一个文生图评估表，覆盖质量、一致性、安全和多样性。
9. 写一个 0 依赖 Python demo，输入分辨率、VAE scale、采样步数、CFG scale、prompt token 数和 ControlNet 条件通道数，输出 latent shape、压缩比、U-Net 调用次数和 gate pass。

## 7.23 本章总结

Stable Diffusion 是 latent diffusion 的代表，它把文生图拆成 text encoder、latent denoiser、scheduler 和 VAE decoder 等模块。DALL·E 系列代表了从自回归图像 token 到更强文生图系统的演进。理解这些系统时，要抓住几个关键点：

1. Latent diffusion 用 VAE 降低扩散空间成本。
2. Text encoder 把 prompt 转成条件表示。
3. U-Net 或 DiT 负责根据条件去噪。
4. Scheduler 控制采样路径。
5. CFG 和 negative prompt 影响 prompt 遵循和图像质量。
6. ControlNet、inpainting、image-to-image 提升可控编辑能力。
7. 自回归图像生成和 diffusion 是两条不同但相互影响的路线。

下一章会进入视频生成与 world model，讲清视频相对图像多出的时间维度、帧间一致性、运动建模、物理世界模拟和视频生成评估。
