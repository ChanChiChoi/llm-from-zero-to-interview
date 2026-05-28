# 第七章：Stable Diffusion 与 DALL·E

上一章讲了 diffusion 的基础：加噪、去噪、scheduler、U-Net、条件生成和 classifier-free guidance。本章进入主流文生图系统，重点讲 Stable Diffusion 和 DALL·E。它们代表了两条重要路线：Stable Diffusion 以 latent diffusion 为核心，通过 VAE、text encoder、U-Net/DiT 和 scheduler 生成图像；DALL·E 系列则经历了从离散图像 token 自回归生成到更强的文生图系统演进。

本章目标不是复现完整模型，而是讲清一个文生图系统由哪些模块组成、prompt 如何影响生成、negative prompt 和 CFG 如何工作、ControlNet 和图像编辑为什么重要，以及自回归图像生成和 diffusion 图像生成的差异。

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

## 7.20 面试官会怎么问

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

## 7.21 小练习

1. 画出 Stable Diffusion 的训练和推理流程。
2. 解释 VAE encoder 和 decoder 分别做什么。
3. 解释 text encoder 如何影响 prompt 遵循。
4. 比较 CFG scale 过低和过高的效果。
5. 设计一个 image-to-image 任务，并说明 denoising strength 的作用。
6. 说明 ControlNet 和普通 prompt 控制的区别。
7. 比较 diffusion 和自回归图像生成。
8. 设计一个文生图评估表，覆盖质量、一致性、安全和多样性。

## 7.22 本章总结

Stable Diffusion 是 latent diffusion 的代表，它把文生图拆成 text encoder、latent denoiser、scheduler 和 VAE decoder 等模块。DALL·E 系列代表了从自回归图像 token 到更强文生图系统的演进。理解这些系统时，要抓住几个关键点：

1. Latent diffusion 用 VAE 降低扩散空间成本。
2. Text encoder 把 prompt 转成条件表示。
3. U-Net 或 DiT 负责根据条件去噪。
4. Scheduler 控制采样路径。
5. CFG 和 negative prompt 影响 prompt 遵循和图像质量。
6. ControlNet、inpainting、image-to-image 提升可控编辑能力。
7. 自回归图像生成和 diffusion 是两条不同但相互影响的路线。

下一章会进入视频生成与 world model，讲清视频相对图像多出的时间维度、帧间一致性、运动建模、物理世界模拟和视频生成评估。
