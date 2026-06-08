# 第八章：视频生成与 World Model

视频生成可以看成图像生成的自然扩展，但它远不只是“连续生成很多张图片”。视频多了时间维度，因此模型不仅要生成每一帧的视觉质量，还要保持帧间一致、动作连贯、物体身份稳定、摄像机运动合理，并尽量符合物理世界规律。这也是为什么视频生成经常和 world model 联系在一起：一个强视频模型可能不只是会画图，还在某种程度上学习了世界如何随时间演化。

本章目标是建立视频生成的主线：视频数据表示、时空 token、video diffusion、时序一致性、文生视频、图生视频、Sora 类路线、world model 直觉、物理一致性、评估和安全风险。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，重点核对了 Video Diffusion Models、Imagen Video、Make-A-Video、Phenaki、Lumiere、VideoPoet、Sora 公开技术说明、World Models 和 Dreamer / latent dynamics 相关资料，以及 FVD 等视频生成评估指标。正文只吸收公开、稳定、适合面试表达的共识，不把任何闭源视频模型的未公开训练配方、数据来源、模型规模或安全策略写成确定事实。

本章聚焦：

1. 视频相对图像多出的时间维度，以及为什么单帧质量不能代表视频质量。
2. 视频张量、帧级 patch、spatiotemporal patch、latent video token 和成本估算。
3. video diffusion 的训练目标、CFG 调用成本、时空 denoiser 和视频条件控制。
4. temporal consistency、identity drift、flicker、motion smoothness、object permanence 和物理一致性。
5. Sora 类路线的公开抽象：patch / latent 统一表示、大规模时空生成模型、长视频和世界模拟候选方向。
6. world model 的状态转移直觉，以及“会生成看似合理视频”与“可交互、可规划、可验证 world model”之间的边界。
7. 可手算的视频 token / 时序一致性审计 demo。

本章不展开完整视频模型训练系统、真实视频采集治理、视频压缩 codec 训练、3D U-Net / DiT 代码实现、光流算法、自动驾驶或机器人控制算法，也不提供 deepfake 制作、身份仿冒或绕过视频安全审核的方法。

## 8.1 视频为什么比图像难

图像是二维空间信号，视频是空间加时间信号。

图像生成关注：

1. 单帧质量。
2. 语义是否符合 prompt。
3. 构图和风格。
4. 细节是否自然。

视频生成还要关注：

1. 帧间一致性。
2. 运动是否连续。
3. 物体身份是否稳定。
4. 摄像机运动是否合理。
5. 物理规律是否可信。
6. 长时间依赖是否保持。

例如一个人走路的视频，如果每帧单独看都很好，但脸一会儿变形、衣服颜色跳变、手的位置闪烁、背景结构漂移，整体视频仍然是失败的。

面试回答：

```text
视频生成比图像生成难，因为它不仅要求每帧图像质量高，还要求时间维度上一致。模型要保持物体身份、背景、动作轨迹、摄像机运动和物理关系的连续性。很多失败不是单帧质量差，而是帧间闪烁、动作不连贯或物理不一致。
```

## 8.2 视频数据如何表示

视频可以表示为帧序列：

```text
video: [T, H, W, C]
```

在 PyTorch 中常见为：

```text
video_tensor: [B, C, T, H, W]
```

其中：

1. `B` 是 batch size。
2. `C` 是通道数。
3. `T` 是帧数。
4. `H` 和 `W` 是分辨率。

视频比图像贵得多，因为 token 数随时间、宽度和高度一起增长。

例如：

```text
16 frames * 32 * 32 spatial patches = 16384 tokens
```

这比单张图像的 token 数大很多。

## 8.3 Spatiotemporal Patch

图像 ViT 把图片切成空间 patch。视频模型可以把视频切成时空 patch。

空间 patch：

```text
patch: [P, P, C]
```

时空 patch：

```text
patch: [T_p, P, P, C]
```

也就是一个 token 不只覆盖一个空间区域，还覆盖连续几帧。

优点：

1. 降低 token 数。
2. 让 token 内部包含短期运动信息。
3. 适合 Transformer 建模长程时空关系。

缺点：

1. 太大的时空 patch 会丢细节。
2. 对快速运动和小目标不友好。
3. token 数仍然很大。

## 8.4 Video Diffusion

视频 diffusion 的直觉和图像 diffusion 类似：

1. 训练时给视频 latent 加噪。
2. 模型学习预测噪声或去噪方向。
3. 生成时从随机视频噪声开始逐步去噪。

区别在于输入输出是视频张量或视频 latent：

```text
noisy video latent + timestep + text condition -> denoised video latent
```

模型需要同时建模：

1. 空间结构。
2. 时间变化。
3. 文本条件。
4. 运动连续性。

Video diffusion 可以用 3D U-Net、时空 Transformer 或二者混合结构实现。

### 8.4.1 关键公式与视频生成速查

**视频张量和 token 数**

视频可以写成：

```math
X\in\mathbb{R}^{B\times C\times T\times H\times W}
```

其中 `B` 是 batch size，`C` 是通道数，`T` 是帧数，`H,W` 是空间分辨率。

如果逐帧按空间 patch 切分，空间 patch size 为 `P`，视频 token 数约为：

```math
N_{\mathrm{frame}}=
T
\left\lceil\frac{H}{P}\right\rceil
\left\lceil\frac{W}{P}\right\rceil
```

如果用时空 patch，时间 patch size 为 `P_t`，空间 patch size 为 `P`，token 数变成：

```math
N_{\mathrm{st}}=
\left\lceil\frac{T}{P_t}\right\rceil
\left\lceil\frac{H}{P}\right\rceil
\left\lceil\frac{W}{P}\right\rceil
```

这解释了为什么视频模型经常需要时空压缩：时间维度会把图像 token 成本再乘上帧数。

**视频 latent 压缩**

视频 diffusion 通常不会直接在原始像素上建模，而是在视频 latent 中去噪。设时间压缩倍数为 `f_t`，空间压缩倍数为 `f_s`，latent channel 为 `C_z`：

```math
Z\in
\mathbb{R}^{B\times C_z\times T/f_t\times H/f_s\times W/f_s}
```

元素量比例可粗略写为：

```math
R_{\mathrm{video}}=
\frac{
C_z(T/f_t)(H/f_s)(W/f_s)
}{
3THW
}
```

这里忽略了 padding、ceil、codec 细节和模型内部 feature map，只用于面试时解释“为什么要 latent video token”。

**Video diffusion 训练目标**

设干净视频 latent 为 `z_0`，加噪后为 `z_t`，文本、图片或姿态等条件为 `c`，denoiser 预测噪声：

```math
\hat{\epsilon}=
\epsilon_\theta(z_t,t,c)
```

常见噪声预测目标：

```math
L_{\mathrm{video}}=
\mathbb{E}_{z_0,t,\epsilon,c}
\left[
\left\|
\epsilon-
\epsilon_\theta(z_t,t,c)
\right\|_2^2
\right]
```

和图像 diffusion 相比，区别不是 loss 形式本身，而是 `z_t` 多了时间维度，denoiser 必须同时解释空间结构和时间变化。

**时空注意力成本**

如果把视频 token 全部做 full self-attention，attention cell 规模约为：

```math
C_{\mathrm{attn}}\approx N_{\mathrm{st}}^2
```

如果每一步 denoising 都有文本 cross-attention，文本 token 数为 `N_c`，采样步数为 `S`，视频 latent token 数为 `N_z`，cross-attention cell 规模约为：

```math
C_{\mathrm{cross}}\approx S N_z N_c
```

CFG 打开时，每个采样步通常需要正条件和负条件两次 denoiser 预测，因此 denoiser 调用数近似为：

```math
C_{\mathrm{denoise}}\approx 2S
```

**Temporal consistency 指标直觉**

视频质量不能只看单帧。一个简单时序一致性审计可以看：

```math
F_{\mathrm{flicker}}=
\frac{1}{T-1}
\sum_{t=2}^{T}
\left|
b_t-b_{t-1}
\right|
```

其中 `b_t` 是第 `t` 帧亮度、颜色或特征统计量。越大说明越可能闪烁。

身份稳定性可以用相邻帧主体 embedding 的最小余弦相似度：

```math
S_{\mathrm{id}}=
\min_{t=2,\ldots,T}
\frac{
e_t^\top e_{t-1}
}{
\|e_t\|_2\|e_{t-1}\|_2
}
```

运动平滑度可以用主体中心位置的二阶差分：

```math
M_{\mathrm{smooth}}=
\frac{1}{T-2}
\sum_{t=2}^{T-1}
\left|
p_{t+1}-2p_t+p_{t-1}
\right|
```

二阶差分过大，往往说明运动突然加速、跳变或跟踪不稳定。

**World model 状态转移**

world model 的最小数学抽象是状态转移：

```math
s_{t+1}\sim p_\theta(s_{t+1}\mid s_t,a_t)
```

如果只是被动视频生成，可以把动作 `a_t` 弱化为文本条件、相机运动或隐式动态条件；如果是机器人或自动驾驶，`a_t` 就是可执行动作。一个可验证 world model 至少要能做 rollout：

```math
\hat{s}_{t+k}=F_\theta^{(k)}(s_t,a_{t:t+k-1})
```

并在预测误差、任务成功率和安全约束上接受评估。面试时要强调：视频生成能学习部分动态规律，但不等于已经具备可控、可规划的世界模型。

**视频评估：FVD 直觉**

FVD 通常把真实视频和生成视频映射到视频特征空间，再比较两个高斯分布的均值和协方差：

```math
\mathrm{FVD}=
\|\mu_r-\mu_g\|_2^2+
\mathrm{Tr}
\left(
\Sigma_r+
\Sigma_g-
2(\Sigma_r\Sigma_g)^{1/2}
\right)
```

它能反映分布级视频质量，但不能单独覆盖 prompt 遵循、物理一致性、身份保持、安全性或具体业务目标。因此视频评估通常需要自动指标、任务型评估和人工偏好结合。

## 8.5 文生视频和图生视频

### 8.5.1 文生视频

输入 prompt，生成视频。

```text
prompt: 一只小狗在雪地里奔跑
output: 多秒视频
```

难点：prompt 不只要控制物体和风格，还要控制动作、镜头、节奏和场景变化。

### 8.5.2 图生视频

输入一张图片，让图片动起来。

```text
image + prompt: 让海面缓慢起伏
output: 动态视频
```

图生视频难点：

1. 保持原图主体身份。
2. 控制运动幅度。
3. 避免背景漂移。
4. 避免物体形变。

## 8.6 时序一致性

时序一致性是视频生成的核心。

常见问题：

1. Flickering：画面闪烁。
2. Identity drift：人物或物体身份漂移。
3. Texture drift：纹理和颜色逐帧变化。
4. Geometry drift：结构逐渐变形。
5. Motion discontinuity：动作不连续。

解决方向：

1. 使用时序 attention。
2. 使用 3D 卷积或时空 Transformer。
3. 使用光流、深度或姿态条件。
4. 用更长视频数据训练。
5. 加入一致性损失或视频判别器。

## 8.7 运动建模

视频中的运动包括：

1. 物体运动。
2. 人体动作。
3. 摄像机运动。
4. 光照变化。
5. 场景变化。

好的视频模型要区分：

1. 主体在动。
2. 镜头在动。
3. 背景在动。
4. 多个物体相互作用。

例如“镜头向前推进”和“物体向镜头走来”在视觉上相似，但语义不同。模型如果没有足够世界理解，可能生成不合理运动。

## 8.8 Sora 类路线的直觉

Sora 类模型的公开信息强调：把视频和图像统一表示为 patches，并在大规模数据上训练生成模型，学习从文本条件到时空内容的映射。

从技术直觉看，关键点包括：

1. 统一处理不同时长、分辨率和宽高比的视频。
2. 使用时空 patch 或 latent token 表示视频。
3. 用大规模 transformer/diffusion 模型建模时空 token。
4. 从大规模视频数据中学习物体、运动和场景规律。
5. 支持文生视频、图生视频和视频延展。

面试时不要编造未公开细节。可以回答公开层面的抽象：Sora 类路线说明视频生成正在从短 clip 生成走向更长时长、更强物理一致性和更统一的世界模拟。

## 8.9 World Model 是什么

World model 可以理解为对世界状态和动态规律的内部建模。

它不仅要知道“物体长什么样”，还要知道：

1. 物体会如何运动。
2. 动作会导致什么后果。
3. 遮挡后物体仍然存在。
4. 重力、碰撞和接触关系。
5. 场景随时间如何变化。

视频生成模型如果能生成长时间、物理一致的视频，就说明它可能学到了一部分世界动态规律。但生成视频看起来合理，不等于它具备完整、可验证、可规划的 world model。

面试回答：

```text
World model 指模型对世界状态和动态规律的内部表示，例如物体持久性、运动、接触、遮挡、因果后果和物理约束。视频生成模型可能通过预测未来帧学习部分世界规律，但会生成合理视频不等于具备完整可控的世界模型，还需要规划、交互和可验证评估。
```

## 8.10 物理一致性

视频生成中的物理一致性包括：

1. 物体不会突然消失。
2. 人体动作符合关节结构。
3. 液体、烟雾、布料运动合理。
4. 光照和阴影一致。
5. 碰撞和接触可信。
6. 镜头运动符合透视。

常见失败：

1. 手穿过物体。
2. 球滚动方向不合理。
3. 人走路脚不接地。
4. 物体遮挡后形状变化。
5. 水流和烟雾运动异常。

物理一致性评估很难，因为它往往需要时间、三维结构和因果关系判断。

## 8.11 长视频生成难点

短视频生成已经难，长视频更难。

难点：

1. 长期身份保持。
2. 故事情节连续。
3. 场景和空间结构保持。
4. 计算成本随时长增长。
5. 训练数据质量和版权问题。
6. 评估成本高。

长视频生成可能需要分层建模：先生成脚本或场景计划，再生成镜头，再生成具体帧。

## 8.12 视频生成的条件控制

视频生成可以被多种条件控制：

1. 文本 prompt。
2. 初始图片。
3. 末帧图片。
4. 姿态序列。
5. 深度序列。
6. 草图或边缘。
7. 摄像机轨迹。
8. 音频。

条件越丰富，控制越强，但数据和模型复杂度也越高。

例如人物跳舞视频，可以使用姿态序列控制动作，用文本控制服装和风格，用初始图控制人物身份。

## 8.13 视频生成评估

视频评估比图像更复杂。

评估维度：

1. 单帧画质。
2. 文本一致性。
3. 时序一致性。
4. 运动自然度。
5. 物体身份保持。
6. 物理合理性。
7. 多样性。
8. 安全性。

常见方法：

1. 人工偏好评测。
2. CLIP/视频文本相似度。
3. FVD 等视频分布指标。
4. 任务型评估，例如动作识别一致性。
5. 物理/几何专项测试。

单一指标很难覆盖视频质量，通常需要多指标和人工评估结合。

## 8.14 安全风险

视频生成的安全风险比图像更高。

包括：

1. Deepfake。
2. 虚假新闻视频。
3. 名人肖像和声音合成。
4. 暴力或违法内容。
5. 隐私泄露。
6. 版权风格和视频素材问题。
7. 误导性监控或证据伪造。

安全策略：

1. 输入 prompt 审核。
2. 输出视频审核。
3. 水印和 provenance。
4. 人脸和身份保护。
5. 高风险场景拒绝。
6. 数据来源治理。

## 8.15 和自动驾驶、机器人关系

视频生成和 world model 与自动驾驶、机器人有天然联系。

因为这些任务需要预测未来：

1. 车辆下一秒会怎么动。
2. 行人是否会穿过马路。
3. 机器人动作会导致什么结果。
4. 物体被推动后会怎么移动。

但生成好看的视频不等于能做控制。控制系统还需要：

1. 状态估计。
2. 动作规划。
3. 奖励或目标。
4. 安全约束。
5. 实时反馈。

所以视频 world model 可以是基础能力，但不能直接等同于可靠智能体。

## 8.16 最小可运行视频 token / 时序一致性审计 demo

下面这个 demo 不生成视频，只审计视频生成系统最容易被忽略的三个问题：视频 token 成本、latent video diffusion 推理成本、时序一致性和 world model rollout 误差。它只依赖 Python 标准库。

```python
from math import sqrt


def ceil_div(a, b):
    return (a + b - 1) // b


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return sqrt(dot(a, a))


def cosine(a, b):
    denom = norm(a) * norm(b)
    return dot(a, b) / denom if denom else 0.0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def video_token_audit(config, frames):
    raw_shape = (
        config["batch"],
        config["channels"],
        config["frames"],
        config["height"],
        config["width"],
    )
    image_patch_tokens_per_frame = (
        ceil_div(config["height"], config["spatial_patch"])
        * ceil_div(config["width"], config["spatial_patch"])
    )
    framewise_tokens = config["frames"] * image_patch_tokens_per_frame
    st_grid = (
        ceil_div(config["frames"], config["temporal_patch"]),
        ceil_div(config["height"], config["spatial_patch"]),
        ceil_div(config["width"], config["spatial_patch"]),
    )
    spatiotemporal_tokens = st_grid[0] * st_grid[1] * st_grid[2]

    latent_shape = (
        config["batch"],
        config["latent_channels"],
        ceil_div(config["frames"], config["temporal_scale"]),
        ceil_div(config["height"], config["spatial_scale"]),
        ceil_div(config["width"], config["spatial_scale"]),
    )
    pixel_elements = (
        config["batch"]
        * config["channels"]
        * config["frames"]
        * config["height"]
        * config["width"]
    )
    latent_elements = 1
    for dim in latent_shape:
        latent_elements *= dim
    latent_tokens = latent_shape[2] * latent_shape[3] * latent_shape[4]
    cfg_calls = config["steps"] * (2 if config["guidance_scale"] > 1.0 else 1)
    cross_attention_cells = config["steps"] * latent_tokens * config["prompt_tokens"]

    centers = [item["center"] for item in frames]
    second_diffs = [
        centers[i + 1] - 2 * centers[i] + centers[i - 1]
        for i in range(1, len(centers) - 1)
    ]
    motion_smoothness = mean(abs(v) for v in second_diffs)
    identity_cosines = [
        cosine(frames[i - 1]["identity"], frames[i]["identity"])
        for i in range(1, len(frames))
    ]
    brightness_diffs = [
        abs(frames[i]["brightness"] - frames[i - 1]["brightness"])
        for i in range(1, len(frames))
    ]
    object_permanence = sum(1 for item in frames if item["visible"]) / len(frames)

    velocity = frames[1]["center"] - frames[0]["center"]
    rollout = [round(frames[0]["center"] + i * velocity, 3) for i in range(len(frames))]
    rollout_mae = mean(abs(pred - gold) for pred, gold in zip(rollout, centers))

    report = {
        "raw_shape": raw_shape,
        "framewise_tokens": framewise_tokens,
        "spatiotemporal_grid": st_grid,
        "spatiotemporal_tokens": spatiotemporal_tokens,
        "token_reduction": round(spatiotemporal_tokens / framewise_tokens, 3),
        "latent_shape": latent_shape,
        "latent_element_ratio": round(latent_elements / pixel_elements, 6),
        "latent_tokens": latent_tokens,
        "denoiser_calls": cfg_calls,
        "cross_attention_cells": cross_attention_cells,
        "motion_smoothness": round(motion_smoothness, 4),
        "min_identity_cosine": round(min(identity_cosines), 4),
        "avg_flicker": round(mean(brightness_diffs), 4),
        "object_permanence": round(object_permanence, 3),
        "world_rollout": rollout,
        "world_rollout_mae": round(rollout_mae, 4),
    }
    checks = {
        "st_tokens_fit_budget": spatiotemporal_tokens <= config["token_budget"],
        "latent_ratio_small": report["latent_element_ratio"] < 0.01,
        "cfg_calls_expected": cfg_calls == config["steps"] * 2,
        "identity_stable": report["min_identity_cosine"] > 0.98,
        "low_flicker": report["avg_flicker"] < 0.04,
        "smooth_motion": report["motion_smoothness"] < 0.03,
        "object_persistent": object_permanence == 1.0,
        "rollout_error_small": report["world_rollout_mae"] < 0.02,
    }
    report["checks"] = checks
    report["gate_pass"] = all(checks.values())
    return report


config = {
    "batch": 1,
    "channels": 3,
    "frames": 16,
    "height": 256,
    "width": 256,
    "spatial_patch": 16,
    "temporal_patch": 2,
    "temporal_scale": 4,
    "spatial_scale": 8,
    "latent_channels": 4,
    "steps": 24,
    "guidance_scale": 6.0,
    "prompt_tokens": 14,
    "token_budget": 4096,
}
frames = [
    {"center": 0.10, "identity": [1.00, 0.00], "brightness": 0.50, "visible": True},
    {"center": 0.21, "identity": [0.995, 0.05], "brightness": 0.52, "visible": True},
    {"center": 0.31, "identity": [0.990, 0.08], "brightness": 0.51, "visible": True},
    {"center": 0.43, "identity": [0.985, 0.10], "brightness": 0.53, "visible": True},
    {"center": 0.54, "identity": [0.980, 0.12], "brightness": 0.52, "visible": True},
]

report = video_token_audit(config, frames)
for key, value in report.items():
    print(f"{key}={value}")
```

一组可核对输出如下：

```text
raw_shape=(1, 3, 16, 256, 256)
framewise_tokens=4096
spatiotemporal_grid=(8, 16, 16)
spatiotemporal_tokens=2048
token_reduction=0.5
latent_shape=(1, 4, 4, 32, 32)
latent_element_ratio=0.005208
latent_tokens=4096
denoiser_calls=48
cross_attention_cells=1376256
motion_smoothness=0.0133
min_identity_cosine=0.9987
avg_flicker=0.015
object_permanence=1.0
world_rollout=[0.1, 0.21, 0.32, 0.43, 0.54]
world_rollout_mae=0.002
checks={'st_tokens_fit_budget': True, 'latent_ratio_small': True, 'cfg_calls_expected': True, 'identity_stable': True, 'low_flicker': True, 'smooth_motion': True, 'object_persistent': True, 'rollout_error_small': True}
gate_pass=True
```

这个 demo 把视频生成的几个面试点变成了可解释数字：

1. `16` 帧视频逐帧切 `16 x 16` patch 有 `4096` 个 token，用 `2` 帧时空 patch 后降到 `2048`。
2. 视频 latent shape 是 `(1, 4, 4, 32, 32)`，元素量只有原始 RGB 视频的约 `0.005208`。
3. `24` 个采样步打开 CFG 后通常对应 `48` 次 denoiser 调用。
4. 时序一致性不能只看美观，要看 flicker、identity cosine、motion smoothness 和 object permanence。
5. world model 的入门审计可以先看简单 rollout 误差，但真实系统还要评估动作条件、因果后果和安全边界。

## 8.17 面试官会怎么问

### 问题一：视频生成为什么比图像生成难？

回答模板：

```text
视频比图像多了时间维度。模型不仅要保证每帧质量，还要保持帧间一致、物体身份稳定、动作连续、镜头运动合理和物理关系可信。很多失败来自闪烁、身份漂移、纹理变化和运动不连续。
```

### 问题二：什么是 spatiotemporal patch？

回答模板：

```text
图像 patch 只覆盖空间区域，spatiotemporal patch 同时覆盖一小段时间和空间区域，也就是 [T_p, P, P, C]。它把视频切成时空 token，便于 Transformer 或 diffusion 模型统一建模空间和时间关系。
```

### 问题三：视频 diffusion 和图像 diffusion 有什么区别？

回答模板：

```text
核心加噪去噪思想相同，但视频 diffusion 的样本包含时间维度，模型要预测视频 latent 的噪声，并同时建模空间结构和时间变化。因此需要时序 attention、3D 卷积或时空 Transformer 来保持运动和帧间一致性。
```

### 问题四：World model 和视频生成有什么关系？

回答模板：

```text
视频生成要求模型预测世界随时间变化，因此可能学到部分物体持久性、运动、遮挡和物理规律。但能生成合理视频不等于具备完整 world model。真正的 world model 还需要可交互、可预测行动后果，并在任务中可验证。
```

### 问题五：如何评估视频生成模型？

回答模板：

```text
要同时评估单帧画质、prompt 一致性、时序一致性、运动自然度、身份保持、物理合理性、安全性和多样性。自动指标很难完全覆盖，通常需要视频指标、任务型评估和人工偏好结合。
```

## 8.18 小练习

1. 比较图像生成和视频生成的主要难点。
2. 给定 `T=16`、空间 patch 为 `32 x 32`，计算视频 token 数。
3. 列出 5 类视频时序不一致 bad case。
4. 设计一个图生视频任务，并说明如何保持主体身份。
5. 解释 Sora 类路线为什么会被称为 world simulator 的候选方向。
6. 设计一个视频生成评估表，覆盖画质、一致性、运动和安全。
7. 分析为什么生成视频不等于能做机器人控制。
8. 写一个 0 依赖视频 token / 时序一致性审计 demo，输出 framewise tokens、spatiotemporal tokens、latent shape、CFG 调用次数、flicker、identity stability、motion smoothness、object permanence 和 rollout MAE。

## 8.19 本章总结

视频生成是多模态生成的重要前沿。它比图像生成多了时间维度，因此核心挑战从单帧质量扩展到时序一致、动作连续、身份保持和物理合理。Sora 类路线说明大规模时空生成模型可能学习到部分世界动态规律，但 world model 是否真正可靠，还需要可交互、可验证和可规划的评估。

需要记住：

1. 视频数据是空间加时间信号。
2. Spatiotemporal patch 把视频切成时空 tokens。
3. Video diffusion 在视频 latent 上做加噪和去噪。
4. 时序一致性是视频生成核心难点。
5. World model 强调世界状态和动态规律建模。
6. 物理一致性、长视频和可控生成仍然很难。
7. 视频生成安全风险包括 deepfake、虚假视频和身份滥用。

下一章会进入语音与音频生成，讲清 ASR、TTS、speech-to-speech、音频 token、codec language model 和语音多模态系统。
