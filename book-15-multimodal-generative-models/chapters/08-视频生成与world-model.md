# 第八章：视频生成与 World Model——从像素序列到可交互的动态模型

上一章把扩散模型放进了文生图系统。本章继续沿着同一条线向时间维度推进：一张图片描述一个时刻的空间状态，一段视频描述状态如何随时间变化。这个变化看起来只是多了一个维度，实际却改变了表示、计算、训练目标、评估方法和系统边界。

如果把每一帧单独生成，再把它们拼接起来，画面可能在每个时刻都漂亮，却会出现闪烁、身份漂移、物体忽隐忽现和运动跳变。视频模型要学习的不是“很多张好看的图片”，而是一个条件分布：在给定文本、初始画面、动作或摄像机轨迹时，生成一条在空间上可信、在时间上连续的信号。

World model 又是另一层概念。它通常指模型内部存在一个能够表示环境状态、预测状态变化，并在必要时接受动作条件的动态模型。视频生成模型可能从视频中学到某些动态规律，但“能生成看起来合理的视频”与“能在行动前预测后果并支持规划”之间仍有很长的距离。本章始终把这两个概念放在一起比较，却不把它们混为一谈。

本章的学习目标有四个层次：

1. 能从帧率、分辨率、patch 和 latent 压缩计算视频的基本成本。
2. 能解释 video diffusion、离散视频 token、自回归模型和时空 Transformer 的关系。
3. 能把时序一致性、运动、物体持久性和物理合理性拆成可观测的评估对象。
4. 能判断一个系统究竟是视频生成器、视频预测器，还是具备动作条件和规划闭环的 world model。

## 8.0 研究对象、资料层级与证据边界

### 8.0.1 三个容易混淆的对象

本章先区分三个对象。

视频生成器从噪声、文本或参考视频生成一段视频。它的主要任务是采样出符合条件的视觉序列，常见目标是文生视频、图生视频、视频编辑和视频延展。

视频预测器从已经观察到的前几帧预测未来帧，条件通常是历史观测，有时还包括动作。它可以是生成式的，也可以只在 latent 空间预测未来表示。预测器不一定支持从任意文本创作视频。

World model 更关注状态和转移规律。它需要回答“当前环境处于什么状态”“采取动作后可能到达什么状态”“这个预测是否足以支持任务决策”。一个 world model 可以用图像、视频、状态向量或压缩 latent 表示世界，不要求最终输出一定是像素级视频。

因此，下面三句话的证据强度不同：

- “模型能根据文字生成视频”是生成能力声明。
- “模型能根据历史帧预测未来”是预测能力声明。
- “模型能根据动作预测结果，并用预测结果选动作”才接近控制型 world model 的定义。

### 8.0.2 资料为什么要分层

视频生成的公开材料经常同时包含论文、产品页面、演示视频、模型卡和媒体报道。它们支持的结论不一样。本章采用下面的证据层级：

| 资料类型 | 可以支持的结论 | 不能直接支持的结论 |
| --- | --- | --- |
| 原始论文 | 论文作者公开的表示、训练目标、实验设置和结果 | 论文之外版本的全部行为 |
| 官方技术说明 | 产品公开的能力范围、示例和安全说明 | 未公开的参数量、数据配方和内部模块 |
| 官方代码或模型卡 | 某个版本的接口、权重和推理配置 | 所有平台部署后的质量和延迟 |
| 评估论文与 benchmark | 指标定义、测试维度和公开实验 | 指标等价于人类整体体验 |
| 教学 demo | 形状、算术和简化模型的可复核结论 | 真实视频的物理真实性和产品可靠性 |
| 新闻或二次解读 | 线索和时间背景 | 关键架构事实和性能保证 |

本章核验了 [Video Diffusion Models](https://arxiv.org/abs/2204.03458)、[Imagen Video](https://arxiv.org/abs/2210.02303)、[Make-A-Video](https://arxiv.org/abs/2209.14792)、[Phenaki](https://arxiv.org/abs/2210.02399)、[Lumiere](https://arxiv.org/abs/2401.12945)、[VideoPoet](https://arxiv.org/abs/2312.14125)、[CogVideoX](https://arxiv.org/abs/2408.06072)、[World Models](https://arxiv.org/abs/1803.10122)、[DreamerV3](https://arxiv.org/abs/2301.04104)、[V-JEPA 2](https://arxiv.org/abs/2506.09985)、[FVD](https://arxiv.org/abs/1812.01717) 和 [VBench](https://arxiv.org/abs/2311.17982) 的论文入口。Sora 的公开资料使用 [OpenAI 官方技术说明](https://openai.com/index/sora/) 作为产品层证据；本章只引用其公开的 patch 化视频表示和能力展示，不推断未公开的训练细节。

### 8.0.3 初学者地图与专家地图

初学者可以先记住一条数据流：

~~~text
文本、图片、历史视频或动作
    -> 条件表示
    -> 视频 latent 或视频 token
    -> 时空生成 / 预测模型
    -> 解码器或视频渲染器
    -> 视频与质量、安全检查
~~~

专家需要把这条数据流拆成更细的契约：

- 帧率和时间戳是否均匀；
- 输入视频使用 [B,C,T,H,W] 还是 [B,T,H,W,C]；
- 空间和时间压缩是否使用向上取整、padding 或 causal 编码；
- denoiser 预测的是 epsilon、x_0 还是 velocity；
- 条件是拼接到 token 序列、进入 cross-attention，还是通过 adapter 注入；
- 长视频是否一次性生成，还是分块、滑窗或层级生成；
- 评估究竟测单帧质量、时间一致性、动作后果，还是闭环任务成功率。

## 8.1 视频不是图片列表：时间维度改变了任务

### 8.1.1 从静态信号到动态信号

单张 RGB 图片可以表示为二维空间上的三通道信号。视频在此基础上增加时间索引：

~~~math
X=\{x_1,x_2,\ldots,x_T\},\qquad x_t\in\mathbb{R}^{C\times H\times W}
~~~

其中 T 是帧数，C 是通道数，H 和 W 是帧的高和宽。若帧率为 r 帧每秒，持续时间为 D 秒，则理想化的帧数为：

~~~math
T\approx rD.
~~~

例如，一段 5 秒、24 fps 的视频约有 120 帧。如果分辨率是 1280 x 720，原始 RGB 元素量约为：

~~~math
3\times120\times720\times1280=331{,}776{,}000.
~~~

这个数还没有计算中间激活、模型参数、梯度和多次采样。视频的困难首先是数据量的困难，然后才是语义和物理推理的困难。

### 8.1.2 单帧质量为什么不够

设视频质量由空间质量 Q_s、文本或条件遵循 Q_c、时序一致性 Q_t、运动自然度 Q_m 和物理合理性 Q_p 共同决定。一个业务任务的成功并不是这些分数的简单平均，而更像多个约束的交集：

~~~math
E_{\mathrm{success}}=
E_s\cap E_c\cap E_t\cap E_m\cap E_p.
~~~

这意味着某个维度出现灾难性错误时，其他维度的高分不能抵消它。例如：

- 每帧都清晰，但人物身份每两秒变化一次；
- 动作很流畅，但文字牌匾在每帧变形；
- 镜头很稳定，但球体穿过桌面；
- 文本对象都出现了，但“先拿杯子再放到桌上”的顺序反了。

图像评估通常可以在一张图片上完成，而视频评估必须观察相邻帧、较长时间窗口，有时还需要对动作和因果后果提出问题。

### 8.1.3 时间一致性并不等于静止

一个容易犯的误解是：只要所有帧几乎一样，视频就很稳定。实际上，完全不动的画面可能具有很低的闪烁，却没有完成“奔跑”“倒水”或“镜头推进”等动作条件。

所以时间质量至少包含两个方向：

1. 保持不应改变的属性，例如身份、背景结构和文字内容。
2. 允许并准确表现应当改变的属性，例如位置、姿态、镜头和光照。

真正的时序评估要同时检查“该稳定的是否稳定”和“该变化的是否变化”。

## 8.2 视频数据表示：帧、时间戳与张量布局

### 8.2.1 帧率和时间戳

帧数 T 不等于真实时间信息。若视频的时间戳为 τ_1,...,τ_T，则相邻帧间隔为：

~~~math
\Delta\tau_t=\tau_t-\tau_{t-1}.
~~~

固定帧率假设要求 Δτ_t 近似相同。手机拍摄、网络传输和多传感器采集可能产生可变帧率。如果模型只接收帧序号，却忽略真实时间间隔，它可能把“快速动作的稀疏采样”误解为“慢动作的密集采样”。

工程系统应在数据管道中明确：

- 是否重采样到固定 fps；
- 丢帧和重复帧如何处理；
- 音频与视频是否按统一时钟对齐；
- 训练时的时间裁剪是否保留动作起点和终点。

### 8.2.2 常见张量布局

原始文件中的帧序列常被描述为：

~~~text
video: [T, H, W, C]
~~~

深度学习代码中也常见下面的布局：

~~~text
[B, C, T, H, W]
[B, T, C, H, W]
[B, T, H, W, C]
~~~

B 是 batch size，C 是通道数，T 是帧数，H,W 是空间分辨率。布局本身没有谁绝对正确，但模型的卷积、注意力、位置编码和数据增强必须使用同一个约定。把 [B,T,H,W,C] 直接传给期望 [B,C,T,H,W] 的 3D 卷积，可能不会立刻报错，却会把时间维当成通道维，造成非常难排查的训练失败。

最小的工程契约应包括：shape、dtype、数值范围、时间顺序、fps、padding mask 和有效帧 mask。对于变长视频，T 只是批次中的最大长度，不能把 padding 帧当成真实观测参与 loss。

### 8.2.3 视频裁剪是数据分布的一部分

训练一个视频模型时，随机取一段连续窗口看起来简单，实际上会改变任务分布。若窗口长度为 L，原视频有 T 帧，合法起点通常满足：

~~~math
0\leq k\leq T-L.
~~~

若动作发生在 [a,b]，随机窗口可能完全不包含它，也可能只包含动作中间的一小段。模型因此学到的不是“动作如何发生”，而可能只是背景和局部姿态。

对动作、因果和交互任务，采样策略应记录：窗口覆盖率、动作起点是否可见、前置状态是否保留、末态是否保留，以及相邻窗口是否重叠。数据增强可以改变颜色、裁剪和噪声，但不能无意中破坏时间顺序。

## 8.3 时空 patch 与 token 预算

### 8.3.1 逐帧空间 patch

图像模型可以把每一帧切成 P x P 的空间 patch。若不补齐，且 H,W 能被 P 整除，则每帧 token 数是：

~~~math
N_{\mathrm{frame}}^{(1)}=\frac{H}{P}\frac{W}{P}.
~~~

若不能整除并采用 padding 或边界补齐，应使用向上取整：

~~~math
N_{\mathrm{frame}}=T\left\lceil\frac{H}{P}\right\rceil\left\lceil\frac{W}{P}\right\rceil.
~~~

这里的 N_frame 是整个视频的 token 数。它保留了每一帧的空间细节，但没有在 token 内部显式聚合短时间运动。

### 8.3.2 时空 patch

时空 patch 同时覆盖连续的 P_t 帧和 P x P 空间区域：

~~~math
\mathcal{P}_{i,j,k}\in\mathbb{R}^{C\times P_t\times P\times P}.
~~~

对应的 token 数为：

~~~math
N_{\mathrm{st}}=\left\lceil\frac{T}{P_t}\right\rceil\left\lceil\frac{H}{P}\right\rceil\left\lceil\frac{W}{P}\right\rceil.
~~~

如果 P_t=2，时间方向 token 数大约减半；代价是一个 token 混合了两帧的信息。快速运动、小物体和短暂遮挡可能在这个聚合过程中被平均掉。

### 8.3.3 一次手算

考虑 16 帧、256 x 256 的视频，空间 patch 为 16 x 16：

~~~math
N_{\mathrm{frame}}=16\times16\times16=4096.
~~~

如果时间 patch 为 2：

~~~math
N_{\mathrm{st}}=8\times16\times16=2048.
~~~

这只是 token 数减半，并不代表总计算量一定减半。每个时空 patch 的输入元素也增加了，投影层的通道维、时空位置编码、层数和注意力实现都会影响真实成本。

### 8.3.4 Attention 的平方成本

若对全部 token 做 full self-attention，注意力矩阵的 cell 数量近似为：

~~~math
C_{\mathrm{attn}}\propto N^2.
~~~

从 4096 token 降到 2048 token，理想化的注意力矩阵规模降为原来的四分之一：

~~~math
\frac{2048^2}{4096^2}=\frac14.
~~~

这解释了时空压缩的价值，也解释了为什么过度压缩会损失小目标和快速动作。工程上常用局部注意力、分块注意力、时间/空间分解注意力、稀疏注意力或 latent 压缩，把平方项限制在可承受范围内。

### 8.3.5 token 预算不是能力预算

更少 token 只能说明计算账本更便宜，不能说明语义能力更强。一个 token 预算评估至少要同时记录：

- 视觉 token 数；
- 文本 token 数；
- 条件控制 token 数；
- denoising step 或自回归长度；
- batch size 和并行度；
- 中间激活峰值；
- 解码和后处理成本。

如果只比较 token 数，却没有固定视频时长、分辨率、fps 和采样步数，比较结果很容易失真。

## 8.4 视频 latent：为什么不直接在像素上生成

### 8.4.1 像素空间的成本

原始 RGB 视频可以写为：

~~~math
X\in\mathbb{R}^{B\times3\times T\times H\times W}.
~~~

直接在这个空间进行每一步去噪，会让卷积或注意力处理巨大的时空张量。即使只保存半精度激活，长视频也会迅速超过显存和带宽预算。

因此许多系统先使用视频 VAE 或其他编码器，把视频压缩成 latent：

~~~math
Z=E_\phi(X),\qquad \hat{X}=D_\psi(Z).
~~~

生成模型在 Z 中加噪和去噪，最后通过 D_psi 解码回视频。

### 8.4.2 时空压缩比

设时间压缩倍数为 f_t，空间压缩倍数为 f_s，latent 通道数为 C_z，忽略 padding 和向上取整，则：

~~~math
Z\in\mathbb{R}^{B\times C_z\times T/f_t\times H/f_s\times W/f_s}.
~~~

latent 与原始 RGB 元素量的比例约为：

~~~math
R_{\mathrm{latent}}=
\frac{C_z(T/f_t)(H/f_s)(W/f_s)}{3THW}
=\frac{C_z}{3f_tf_s^2}.
~~~

例如 C_z=4、f_t=4、f_s=8 时：

~~~math
R_{\mathrm{latent}}=\frac{4}{3\times4\times8^2}=\frac1{192}\approx0.005208.
~~~

这只是元素数量的理想化比例，不是显存比例，也不是质量比例。模型还需要 feature map、注意力缓存、解码器激活和临时 buffer。

### 8.4.3 压缩的收益与代价

latent 压缩的收益包括：

1. 减少每一步 denoising 的时空元素数量。
2. 让更大的时间窗口可以放入训练和推理预算。
3. 可以复用图像生成模型中的空间表征和文本条件接口。

代价包括：

1. VAE 重建会丢失细小文字、细线和高频纹理。
2. 时间压缩可能抹掉短暂动作和瞬间接触。
3. 编码器的错误会成为生成模型无法恢复的输入损失。
4. 不同 VAE 的 scaling、时间 padding 和 causal 语义不同，不能只交换权重。

因此视频质量问题可能发生在三个位置：生成器没有生成正确内容，latent decoder 无法还原细节，或者后处理插帧和超分辨率改变了时间一致性。排查时不能只看最终视频。

### 8.4.4 Causal video VAE 的特殊性

若视频要在线生成或逐段延展，编码器和解码器可能需要 causal temporal 结构：当前输出只能依赖当前和过去帧，不能偷看未来。非 causal 编码器可以使用整个窗口，因此离线质量可能更高，但不适合严格的流式延迟约束。

这带来一个容易被忽视的实验差异：离线重建和在线重建并不是同一个任务。系统评估应记录是否允许未来帧、窗口右边界是否 padding，以及延展时是否复用上一窗口的 latent 状态。

## 8.5 Video diffusion：在时空 latent 上学习去噪

### 8.5.1 前向加噪

视频 diffusion 的前向过程与图像 diffusion 形式相同，只是 z_0 现在包含时间维度。令干净视频 latent 为 z_0，时间步为 t，噪声为 epsilon：

~~~math
z_t=\sqrt{\bar{\alpha}_t}z_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).
~~~

bar_alpha_t 决定信号和噪声在第 t 步的比例。z_t 中相邻帧之间原本的运动关系会随着噪声增强而变得难以辨认，denoiser 需要从带噪的时空上下文中恢复它。

### 8.5.2 训练目标

条件可以是文本 c、参考图片 i、历史视频 h、姿态 p、深度 d 或动作 a 的组合。噪声预测器写作：

~~~math
\hat{\epsilon}=\epsilon_\theta(z_t,t,c).
~~~

常见的噪声预测目标为：

~~~math
L_\epsilon=\mathbb{E}_{z_0,t,\epsilon,c}\left[\left\|\epsilon-\epsilon_\theta(z_t,t,c)\right\|_2^2\right].
~~~

也可以预测干净 latent x_0 或 velocity v。关键不是记住某个参数化名称，而是确认训练 target、scheduler 的更新公式和推理时的 prediction type 一致。把预测 epsilon 的模型交给期望 velocity 的 scheduler，系统往往仍然能运行，却会产生严重质量退化。

### 8.5.3 Denoiser 需要同时看空间和时间

图像 U-Net 只需处理二维邻域；视频 denoiser 还要处理：

- 同一帧内相邻区域的空间结构；
- 同一位置跨帧的外观变化；
- 物体移动后新的空间位置；
- 文本或参考图像与时空区域的对应关系。

常见实现路线包括 3D U-Net、二维空间模块加 temporal module、时空 Transformer，以及 latent video 上的 Diffusion Transformer。它们的区别不只是网络名称，还涉及注意力范围、时间位置编码、显存峰值和长视频扩展方式。

### 8.5.4 从图像模型迁移空间知识

[Video Diffusion Models](https://arxiv.org/abs/2204.03458) 展示了把图像和视频数据共同用于扩散训练的路线；[Make-A-Video](https://arxiv.org/abs/2209.14792) 则明确利用大规模文本-图像知识学习“世界长什么样”，再用无文本视频学习“世界如何运动”。这个分工解释了为什么视频模型常从成熟的文生图模型初始化空间模块，再新增或训练时间模块。

迁移的好处是减少视频文本配对数据需求、继承对象和风格知识。风险是图像模型的静态偏置可能被带入视频：模型知道“杯子长什么样”，却不一定知道杯子被推到桌边后会怎样倾倒。

### 8.5.5 CFG 与视频推理成本

若使用 classifier-free guidance，正条件预测和无条件预测通常需要两次 denoiser forward。采样步数为 S 时，理想化调用数约为：

~~~math
C_{\mathrm{denoise}}\approx2S.
~~~

如果每次预测处理 N_z 个视频 latent token，文本条件有 N_c 个 token，cross-attention cell 数量的粗略估计为：

~~~math
C_{\mathrm{cross}}\approx S N_zN_c.
~~~

上式默认只统计一个条件分支。若 CFG 同时执行正条件和负/空条件，且两者 token 数分别为 $N_c^+$ 与 $N_c^-$，则更接近实际的交互量是 $S N_z(N_c^+ + N_c^-)$；即使把两路拼成一个 batch，也只是改变 kernel 的组织方式，不会让第二路条件的 attention cell 凭空消失。

CFG 会提高条件遵循，但过大的 guidance scale 可能造成动作僵硬、纹理重复、颜色过饱和和多样性下降。视频中这种副作用更明显，因为每个时间步的过强条件会把不确定的运动压到少数模式。

### 8.5.6 级联空间与时间超分辨率

[Imagen Video](https://arxiv.org/abs/2210.02303) 采用基础视频生成模型与空间、时间超分辨率模型交替组成的级联路线。它的工程直觉是：先在较低分辨率或较短时空尺度解决语义和大运动，再逐级补充空间细节和帧率。

级联路线把一个难问题拆成多个相对可控的问题，但每个阶段都会引入新的误差。如果基础阶段把手的位置生成错了，后面的超分辨率只会把错误变得更清晰；如果插帧模型不理解遮挡，帧率增加可能带来更多不自然的中间帧。

### 8.5.7 一次生成整段视频与关键帧路线

[Lumiere](https://arxiv.org/abs/2401.12945) 的论文摘要强调 Space-Time U-Net 在单次模型处理路径中生成完整时间长度的视频，而不是先生成相距较远的关键帧再做时间超分辨率。两种路线的核心差异是时间依赖在哪里建立：

- 关键帧路线先解决稀疏时间点，再用插值或超分辨率补全中间帧；
- 全时空路线在较早阶段就让不同时间位置相互作用。

前者可能更容易扩展分辨率和时长，后者更有机会直接约束全局运动；但两者都不能自动保证物理正确。架构描述来自论文，不能扩展为“所有同类系统都采用这一实现”。

## 8.6 其他生成路线：离散 token、自回归与混合架构

### 8.6.1 离散视频 token

另一条路线先训练视频 tokenizer，把连续视频编码为离散 token 序列：

~~~math
z_{1:N}=Q(E_\phi(X)),\qquad \hat{X}=D_\psi(z_{1:N}).
~~~

E_phi 是编码器，Q 是量化器，D_psi 是解码器。生成模型学习条件分布：

~~~math
p(z_{1:N}\mid c)=\prod_{i=1}^{N}p(z_i\mid z_{<i},c).
~~~

优点是可以复用语言模型的 next-token 训练范式，条件也容易统一成 token。代价是长视频 token 数可能非常大，自回归生成延迟随序列长度增长，并且 tokenizer 的量化误差会限制最终视频质量。

### 8.6.2 Phenaki 的变量长度思路

[Phenaki](https://arxiv.org/abs/2210.02399) 公开了用离散视频 token 和带时间因果结构的 tokenizer 处理变量长度视频的路线，并使用时间变化的文本提示生成更长的视频。它的重要启发不是某一个产品数字，而是：如果视频表示支持因果时间结构，模型就有机会把“下一段故事”接在已有视频之后。

变量长度并不等于无限长记忆。长序列仍会受到 token 数、误差积累、身份保持和场景状态容量的限制。每个新窗口都只看最近几帧时，模型可能忘记较早发生的物体关系。

### 8.6.3 VideoPoet 的多模态语言模型路线

[VideoPoet](https://arxiv.org/abs/2312.14125) 论文描述了 decoder-only Transformer 处理图像、视频、文本和音频等多模态输入，并以预训练和任务适配的方式组织目标。它代表一种“把多种生成对象变成可被语言模型处理的序列”的思路。

这种统一路线的价值是接口和任务迁移更灵活；风险是不同模态 token 的时间尺度、压缩误差和解码成本并不相同。把图像 token、音频 token 和视频 token 简单拼在一起，不会自动得到稳定的跨模态时钟。

### 8.6.4 混合路线的比较

| 路线 | 状态空间 | 主要优点 | 主要风险 |
| --- | --- | --- | --- |
| latent video diffusion | 连续视频 latent | 画质和空间迁移能力强 | 采样步数多、时间一致性难 |
| 级联 diffusion | 多级空间/时间 latent | 可逐级控制分辨率和帧率 | 级联误差、延迟和系统复杂度 |
| 离散 token 自回归 | codebook token 序列 | 目标统一、易与 LM 结合 | 序列很长、解码慢、量化误差 |
| 时空 DiT | latent patch/token | 统一时空注意力和扩展接口 | 注意力和显存成本高 |
| latent dynamics/world model | 状态 latent 与转移 | 支持预测、动作条件和规划 | 输出不一定是高保真视频，闭环误差难 |

比较模型时应先问“它在什么状态空间中做什么任务”，再谈参数量和生成时长。

## 8.7 文生视频：文本如何约束动作和镜头

### 8.7.1 文本条件比图像 prompt 多一层时间语义

图像 prompt 主要描述对象、属性、关系、构图和风格。视频 prompt 还要描述：

- 谁在动；
- 动作的起点、方向、速度和终点；
- 摄像机是固定、平移、旋转还是推进；
- 场景变化发生在什么时候；
- 哪些对象应该保持不变。

例如“一个红色纸飞机从桌面左侧飞到右侧，镜头缓慢向前推进”包含至少两个运动主体：纸飞机和摄像机。若没有明确区分，模型可能让桌面移动、纸飞机静止，或者让镜头运动和物体运动叠加成不可解释的结果。

### 8.7.2 文本不能精确表达所有轨迹

文字适合表达语义意图，不擅长给出每一帧的位置约束。对精确动作，应加入姿态序列、深度图、分割 mask、摄像机轨迹、光流或参考视频。文本和结构条件之间可能发生冲突，例如文本要求“手臂抬起”，姿态序列却要求手臂下垂；系统需要规定哪个条件优先，或对冲突样本拒绝生成。

### 8.7.3 文生视频的最低评估样本

一个不应只展示漂亮样片的评估集，至少包含：

1. 单主体平移动作，用于检查运动方向和速度。
2. 关节动作，用于检查人体结构和脚接触。
3. 物体交互，用于检查接触、遮挡和持久性。
4. 摄像机运动，用于区分主体运动和视角运动。
5. 多对象关系，用于检查对象身份和相对位置。
6. 时间顺序指令，用于检查先后关系。

这些样本应把“画面是否漂亮”和“约束是否满足”分开打分。

## 8.8 图生视频与参考视频编辑

### 8.8.1 图生视频的条件分工

图生视频以一张图片作为初始外观锚点，再用文本或运动条件让场景发生变化。可以把条件分成三类：

1. 外观条件：人物、服装、颜色、背景和构图。
2. 运动条件：姿态、轨迹、光流或摄像机路径。
3. 变化范围：哪些区域可以动，哪些区域必须保持。

如果只给初始图片和一句“让它动起来”，模型需要自己猜测运动，身份保持和动作自然度都更不稳定。

### 8.8.2 噪声强度与保持率

图生视频常把输入图片编码成 latent，再在某个噪声时间步开始反向采样。设开始时间步为 t_start，噪声强度越高，生成分布离输入图越远，创造性更强，但输入保持率通常下降。

可以用一个抽象的保持-变化权衡表示：

~~~math
Q_{\mathrm{total}}(s)=\lambda_{\mathrm{keep}}Q_{\mathrm{keep}}(s)+\lambda_{\mathrm{motion}}Q_{\mathrm{motion}}(s),
~~~

其中 s 是 strength，Q_keep 衡量主体和背景保持，Q_motion 衡量运动是否足够，两个权重取决于任务。广告产品更重视主体和文字保持，舞蹈生成更重视动作幅度。

### 8.8.3 视频到视频和局部编辑

视频编辑还需要 mask、参考帧和跨帧约束。单帧 inpainting 逐帧独立处理会产生纹理闪烁；把整段视频一次性送入模型又会提高显存和数据要求。滑窗编辑要处理窗口边界：

- 相邻窗口的重叠区域是否使用相同条件；
- 两个窗口的 latent 如何融合；
- 物体在窗口边缘发生遮挡时，状态如何传递；
- 后处理是否改变音视频同步。

一个成功的编辑系统应同时报告编辑区域的遵循度和非编辑区域的保持率，不能只看编辑区域是否变成目标风格。

## 8.9 条件控制：姿态、深度、摄像机与音频

### 8.9.1 结构条件的价值

文本告诉模型“想要什么”，结构条件告诉模型“在哪里以及如何变化”。常见条件包括姿态序列、深度序列、边缘、分割、光流、摄像机轨迹和音频节拍。

条件通常通过下面三种方式进入模型：

1. 与视频 latent 或 patch embedding 拼接。
2. 经过 encoder 后进入 cross-attention。
3. 通过 adapter、残差分支或控制网络注入中间层。

每种方式都需要定义时间对齐。若视频为 r_v fps、姿态为 r_p fps，映射关系不能只用数组下标相等，而应使用时间戳或明确的重采样策略。

### 8.9.2 条件对齐公式

设视频帧时间为 τ_t，条件序列时间为 σ_j，最简单的最近邻对齐为：

~~~math
j^*(t)=\arg\min_j|\sigma_j-\tau_t|.
~~~

对连续轨迹可以使用线性插值：

~~~math
c(\tau_t)=\frac{\sigma_{j+1}-\tau_t}{\sigma_{j+1}-\sigma_j}c_j+\frac{\tau_t-\sigma_j}{\sigma_{j+1}-\sigma_j}c_{j+1}.
~~~

变量 c_j 可以是二维姿态坐标、深度或摄像机参数。插值对平滑轨迹有帮助，但不能修复原始条件的错误或遮挡。

### 8.9.3 多条件冲突

条件越多不一定越好。文本可能要求“静止的雕像”，摄像机轨迹要求快速推进，姿态条件又要求雕像抬手。系统应定义优先级、置信度或冲突损失，而不能把多个条件无说明地叠加。

一种教学性的加权目标可以写为：

~~~math
L_{\mathrm{cond}}=\lambda_tL_{\mathrm{text}}+\lambda_sL_{\mathrm{structure}}+\lambda_mL_{\mathrm{motion}}.
~~~

lambda_t、lambda_s 和 lambda_m 不是普适的最佳超参数。它们表示不同条件的相对重要性，需要在目标数据和任务上通过消融验证。增加权重可能提高某一条件的遵循，却损伤另一个条件或降低多样性。

## 8.10 时序一致性：应该稳定什么、允许变化什么

### 8.10.1 闪烁

闪烁是相邻帧在不应变化的亮度、颜色或纹理上发生非语义跳变。对帧级统计量 b_t，可以定义一个简单的亮度变化指标：

~~~math
F_{\mathrm{flicker}}=\frac1{T-1}\sum_{t=2}^{T}|b_t-b_{t-1}|.
~~~

它适合做教学和回归测试，不适合单独判断真实闪烁。镜头本来就有闪光、光照变化或快速运动时，b_t 的变化可能是正确的。更可靠的做法是对背景区域、主体区域和全画面分别测量，并用光流或跟踪结果区分真实运动。

### 8.10.2 身份漂移

设第 t 帧中主体的 embedding 为 e_t，相邻帧余弦相似度为：

~~~math
s_t=\frac{e_t^\top e_{t-1}}{\|e_t\|_2\|e_{t-1}\|_2}.
~~~

可以报告平均值、低分位数或最小值：

~~~math
S_{\mathrm{id}}=\min_{t=2,\ldots,T}s_t.
~~~

但 embedding 相似度只反映所选特征空间的相似程度。如果主体检测错位、相邻帧没有对应主体，或者特征编码器本身忽略了服装细节，这个指标都会误导。因此身份保持应配合跟踪成功率、属性准确率和人工抽检。

### 8.10.3 运动平滑度

设主体中心或关键点位置为 p_t。一阶差分近似速度，二阶差分近似加速度：

~~~math
v_t=p_t-p_{t-1},\qquad a_t=p_{t+1}-2p_t+p_{t-1}.
~~~

平均二阶差分可以作为运动跳变的粗略指标：

~~~math
M_{\mathrm{smooth}}=\frac1{T-2}\sum_{t=2}^{T-1}\|a_t\|_2.
~~~

这个指标越小通常越平滑，但“平滑”也不是“自然”。一辆车在转弯或碰撞时确实会有较大的加速度。评估应把物理场景和运动意图纳入解释。

### 8.10.4 纹理、几何与文字保持

时序失败可以按层次拆分：

- 颜色和纹理漂移：物体还在，但表面细节跳变；
- 几何漂移：轮廓、关节或物体结构逐渐变形；
- 关系漂移：物体之间的距离、遮挡和接触关系变化；
- 语义漂移：人物、服装、文字或事件身份改变。

不同层次需要不同观测。像素差对摄像机运动很敏感，光流对遮挡和纹理重复敏感，检测器对小目标和错误检测敏感。一个指标不能替代整套诊断。

### 8.10.5 物体持久性和遮挡

物体被遮挡时，画面中暂时看不见它，并不意味着它从世界中消失。可见性 q_t 可以是二值变量，也可以是检测置信度。一个简单的持久性比例为：

~~~math
P_{\mathrm{visible}}=\frac1T\sum_{t=1}^{T}q_t.
~~~

它只能表示可见，不表示被遮挡后重新出现的物体仍然是同一个物体。更严格的测试应构造“物体被柱子遮挡后再次出现”的样本，比较遮挡前后的颜色、形状、位置和身份 embedding，并检查模型有没有凭空生成第二个物体。

### 8.10.6 主体运动与摄像机运动

“物体向前移动”和“摄像机向物体推进”可能产生相似的像面变化。区分两者需要：

1. 多个背景点的共同运动；
2. 深度或视差线索；
3. 摄像机轨迹条件；
4. 主体相对背景的运动；
5. 任务要求的动作语义。

只用主体中心轨迹，无法判断运动的物理原因。视频模型如果只学到二维表面变化，可能在镜头移动时让背景和主体以同一深度速度运动，导致空间结构不可信。

## 8.11 让时间一致性变好的方法与代价

### 8.11.1 时序注意力和 3D 卷积

时序 attention 让某一帧的表示访问其他帧；3D 卷积在局部时间窗口内聚合空间和时间邻域。两者都能引入跨帧信息，但注意力的成本通常随时间 token 数平方增长，3D 卷积的感受野则受核大小和层数限制。

一种工程折中是空间 attention 与时间 attention 分解：先在每帧内建模空间，再沿同一空间位置或局部窗口建模时间。这样降低计算，但可能弱化快速移动物体跨空间位置的关联。

### 8.11.2 光流、深度和姿态条件

外部运动条件能减少模型自行猜测轨迹的负担。光流提供像素运动，深度提供相对距离，姿态提供人体骨架。它们都是观测或估计，不是绝对真值：光流会在遮挡和无纹理区域失败，深度估计会有尺度和边界误差，姿态检测会漏检关节。

如果把错误条件强制注入模型，输出可能比无条件生成更不自然。因此要把条件置信度传给模型，或在评估中单独测试条件噪声鲁棒性。

### 8.11.3 一致性损失

可在训练目标中加入相邻帧特征或光流 warp 的一致性项。令 W_{t->t+1} 表示把第 t 帧特征按照估计运动变换到下一帧，示意性的损失为：

~~~math
L_{\mathrm{temp}}=\sum_{t=1}^{T-1}\left\|h_{t+1}-W_{t\rightarrow t+1}(h_t)\right\|_1.
~~~

总目标可以写成：

~~~math
L=L_{\mathrm{diff}}+\lambda_{\mathrm{temp}}L_{\mathrm{temp}}.
~~~

这个目标的风险是过度平滑：模型为了降低一致性损失，可能把本来应该变化的纹理和动作压平。lambda_temp 需要通过动作保真度、身份保持和多样性的联合评估选择。

### 8.11.4 长上下文和记忆

增加时间窗口可以让模型看到更长的上下文，但不自动解决长期漂移。窗口变长后，token 数和注意力成本增加；模型还可能把早期细节压缩得过于粗糙。工程系统经常使用摘要 latent、关键帧、对象状态或记忆 token 传递长期信息。

记忆必须区分“可见历史”和“推断状态”。如果模型把一个错误的身份或位置写入记忆，后续窗口会不断重复这个错误。应记录记忆更新规则，并用遮挡、回到旧场景和长时间身份测试它。

## 8.12 长视频：时间长度带来的新问题

### 8.12.1 误差如何累积

假设每个时间步的预测误差平均为 e，模型把自己的预测继续作为下一步输入。最简单的上界直觉是：多步误差随步数累积，可能接近：

~~~math
E_k\lesssim \sum_{i=1}^{k}e_i.
~~~

实际误差可能因为系统稳定性衰减，也可能因为非线性动力学放大；公式不是定理，而是提醒我们不能用 2 秒视频的表现直接外推 2 分钟视频。

### 8.12.2 分块生成与窗口拼接

长视频通常按窗口生成，每个窗口有长度 L，相邻窗口重叠 O。第 j 个窗口的起点可以写为：

~~~math
k_j=j(L-O).
~~~

重叠区域可以使用 latent 融合、关键帧约束、特征匹配或选择一个窗口的结果。拼接失败常表现为：

- 人物在边界处换脸；
- 背景纹理出现接缝；
- 运动速度突然变化；
- 音频与画面时间偏移。

窗口重叠越大，保持上下文的机会越多，但计算重复也越多。必须同时记录有效视频长度和重复计算量。

### 8.12.3 层级生成

故事型长视频可以分成：

1. 事件或脚本层：决定先后关系和角色。
2. 镜头层：决定场景、机位、持续时间和转场。
3. 运动层：决定主体轨迹、姿态和交互。
4. 画面层：生成具体视频帧和细节。

层级生成有利于控制和编辑，但上层计划与下层画面之间会出现语义丢失。若脚本要求“杯子被放在桌面右侧”，镜头规划和视频生成都要保留这个关系；只生成互相漂亮的镜头并不能保证故事连续。

### 8.12.4 长视频的存储和评估成本

评估 10 秒视频和评估 10 分钟视频的成本不只是线性增加。长视频需要人工观看更久，错误可能只在末尾出现，且多段动作的组合会产生新的失败模式。应分开报告：短窗口质量、跨窗口一致性、长程事件保持、单位时长推理成本和人工审核时间。

## 8.13 World model：从生成未来画面到预测状态转移

### 8.13.1 最小状态转移模型

令 s_t 表示时刻 t 的环境状态，a_t 表示动作，o_t 表示观察。一个随机状态转移模型可以写成：

~~~math
s_{t+1}\sim p_\theta(s_{t+1}\mid s_t,a_t),\qquad o_t\sim p_\theta(o_t\mid s_t).
~~~

s_t 不一定是像素，可以是压缩 latent、对象状态或机器人的内部状态。o_t 是摄像头、传感器或视频帧看到的结果。这样写的好处是把“世界如何变化”和“观察如何产生”分开。

对被动视频生成，条件 a_t 可能不存在，或者被文本、摄像机轨迹和隐式运动条件替代。对机器人和自动驾驶，a_t 是实际可执行的转向、速度、抓取或控制信号。

### 8.13.2 视频生成与 world model 的交集

视频生成模型学习的是某种 p(X_future | X_history,c)。如果它在历史视频、条件和长期时间上表现良好，说明内部可能包含部分动态规律。但这仍可能只是观测分布的拟合：

- 它可以生成训练数据中常见的动作，却无法处理未见过的动作组合；
- 它可以让球“看起来像在滚”，却无法预测改变地面坡度后的结果；
- 它可以根据文字生成“机器人拿起杯子”，却不接受真实控制信号；
- 它可以在开放环预测里正确，却在把自身预测作为下一步输入后迅速漂移。

因此“world model”不能只根据输出视频的视觉逼真度命名。

### 8.13.3 表示是否包含控制所需信息

一个用于规划的 latent 状态需要保留任务相关信息。设两个历史观察映射到同一个 latent，却对应不同的可行动后果，表示就出现了状态混叠：

~~~math
E(s_t)=E(s'_t),\qquad p(o_{t+1}\mid s_t,a_t)\ne p(o_{t+1}\mid s'_t,a_t).
~~~

这里 E 是编码器。对生成画面而言，两个状态看起来很像可能没有问题；对控制而言，隐藏的摩擦、物体重量、电量或碰撞关系可能决定动作结果。评估 world model 时必须设计能区分这些状态的任务，而不能只看重建误差。

### 8.13.4 部分可观测环境与 belief state

现实环境通常不是完全可观测的。遮挡、传感器噪声和不可见内部状态使模型只能维护一个 belief state b_t，即对可能状态的分布：

~~~math
b_t(s)=p(s_t=s\mid o_{\leq t},a_{<t}).
~~~

动作后，belief state 先通过转移预测，再结合新观测更新。离散化写法可以表达为：

~~~math
b_{t+1}(s')\propto p(o_{t+1}\mid s')\sum_s p(s'\mid s,a_t)b_t(s).
~~~

这说明“记住最近一帧”不一定等于理解世界。被遮挡的物体、未观察到的速度和隐藏障碍都需要以不确定状态保留，而不是直接猜一个确定答案。

## 8.14 World model 的训练路线

### 8.14.1 像素重建型路线

最直观的路线预测未来像素或解码后的观测。优点是输出可视化，错误容易被人发现；缺点是像素损失会把多个合理未来平均成模糊结果，也会把与任务无关的纹理纳入训练目标。

像素重建适合检查外观和可视化，但不一定能学到可规划的状态。一个机器人只关心“抓取是否成功”，不需要精确预测每根背景草叶的颜色。

### 8.14.2 latent dynamics

latent dynamics 先把观测编码成 z_t，再学习：

~~~math
z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t).
~~~

解码器可只用于可视化，不必参与每一步规划。这样更节省计算，也更容易在任务相关空间中学习，但 latent 的语义可能不透明，且编码器如果丢失了碰撞和接触信息，后续 dynamics 无法补回。

### 8.14.3 JEPA 类预测型表征

JEPA 类方法在表征空间预测被遮挡或未来内容，而不要求逐像素重建。它的优势是可以把训练信号集中在更抽象的结构和时空关系上，减少对纹理细节的依赖。

[V-JEPA 2](https://arxiv.org/abs/2506.09985) 的论文摘要报告了从大规模视频和图像自监督预训练，再用少量机器人视频训练 latent action-conditioned world model 的路线，并展示理解、预测和机器人规划实验。这里可以支持的结论是论文公开了这种训练范式和实验；不能据此断言任意视频生成模型都已经具备同等的机器人规划能力，也不能把论文实验结果替代目标设备上的安全验证。

### 8.14.4 Dreamer 类 imagined rollout

[World Models](https://arxiv.org/abs/1803.10122) 展示了在压缩空间和动态模型中学习环境，并在模型产生的“梦境”中训练策略的早期路线。[DreamerV3](https://arxiv.org/abs/2301.04104) 则代表在学习到的世界模型中想象未来并改进策略的系统。

“在模型内部想象”可以降低真实环境交互成本，但模型错误会污染策略。如果 world model 没有表示某个危险后果，策略可能在 imagined rollout 中表现很好，部署后却失败。真实环境回放、分布外测试和安全约束不可省略。

## 8.15 Rollout、规划与闭环验证

### 8.15.1 多步 rollout

给定初始状态 s_t 和动作序列 a_t,...,a_{t+k-1}，模型可以递推预测：

~~~math
\hat{s}_{t+1}=F_\theta(s_t,a_t),\qquad
\hat{s}_{t+k}=F_\theta(\hat{s}_{t+k-1},a_{t+k-1}).
~~~

第 k 步误差可用真实状态 s_{t+k} 和预测状态比较：

~~~math
E_k=d(\hat{s}_{t+k},s_{t+k}).
~~~

距离 d 要匹配任务。像素 MSE 可能不适合抓取任务，应该同时测物体位置、碰撞、目标达成或控制误差。

### 8.15.2 开放环与闭环

开放环评估把真实历史状态持续提供给模型，只测下一步或固定未来的预测。它可以回答“模型能否预测给定真实历史”，但不测误差反馈。

闭环评估让模型使用自己的预测继续 rollout，或让规划器根据模型预测选择下一个动作，再把动作送入环境。闭环才接近控制系统，但实验成本更高，也更容易暴露累积误差。

最小的评估矩阵应包含：

| 评估方式 | 输入 | 能回答的问题 | 不能回答的问题 |
| --- | --- | --- | --- |
| 单步开放环 | 真实历史与动作 | 短期预测是否准确 | 长期误差是否爆炸 |
| 多步开放环 | 真实起点和动作序列 | 固定 horizon 的预测能力 | 模型误差反馈下的行为 |
| 闭环 rollout | 模型自身预测 | 累积误差和稳定性 | 真实环境安全性仍有限 |
| 规划闭环 | 模型预测加动作选择 | 是否支持任务决策 | 未覆盖的真实风险 |
| 真实环境测试 | 传感器和实际动作 | 部署行为 | 不能仅用少量成功替代全面验证 |

### 8.15.3 模型预测控制

模型预测控制可以在每个时刻生成候选动作序列，利用 world model 预测结果并选择代价较低的一条，只执行第一步，然后重新观察。抽象目标为：

~~~math
\min_{a_{t:t+H-1}}
\mathbb{E}\left[\sum_{k=0}^{H-1}\ell(\hat{s}_{t+k},a_{t+k})\right]
\quad\text{subject to}\quad
\hat{s}_{t+k+1}=F_\theta(\hat{s}_{t+k},a_{t+k}).
~~~

H 是规划 horizon，ell 是任务代价。若模型不确定性重要，应把期望风险、碰撞概率或最坏情况约束加入目标。一个只优化生成画面美观度的 world model 不能直接替代这个控制目标。

## 8.16 物理一致性：视觉合理不等于因果正确

### 8.16.1 物理一致性的层次

视频物理评估可以按层次组织：

1. 几何层：物体轮廓、尺度、遮挡和透视。
2. 运动层：速度、加速度、关节约束和接触。
3. 动力学层：重力、碰撞、摩擦和弹性。
4. 因果层：动作是否产生正确后果，改变条件后结果是否随之改变。

画面看起来连续只说明部分几何和运动约束可能满足，不足以证明动力学或因果层正确。

### 8.16.2 具体失败模式

常见的失败包括：

- 手指穿过杯壁却没有改变轨迹；
- 球滚过桌边后悬空或瞬移；
- 人物脚步移动但脚底没有与地面接触；
- 物体被遮挡后重新出现时质量、形状或颜色改变；
- 液体流动方向与容器倾角不一致；
- 镜头推进时远近物体的相对速度不符合透视；
- 物体被推动后没有反作用，或者反作用方向错误。

这些错误很难用单一 FVD 或图文相似度发现，需要对象级标注、物理规则、人工判断和反事实样本。

### 8.16.3 反事实测试

反事实测试改变一个条件，只观察与该条件相关的结果是否改变。例如：

1. 保持场景和物体不变，只把地面坡度从 0 度改为 10 度。
2. 保持动作不变，只把球的初始位置向左移动。
3. 保持镜头不变，只改变施力方向。
4. 让物体暂时被遮挡，再要求它从另一侧出现。

测试重点不是生成画面是否相似，而是相似的部分应保持、受干预的部分应变化、变化方向应符合任务规则。这种评估比“看一段样片”更接近对 world model 的检验。

## 8.17 视频生成评估：从分布指标到任务指标

### 8.17.1 FVD 的定义和限制

[FVD](https://arxiv.org/abs/1812.01717) 把真实视频和生成视频送入视频特征提取器，在特征空间中分别估计高斯分布，然后计算 Fréchet 距离：

~~~math
\mathrm{FVD}=\|\mu_r-\mu_g\|_2^2+\mathrm{Tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
~~~

mu_r、Sigma_r 是真实样本特征的均值和协方差，mu_g、Sigma_g 是生成样本对应统计量。通常距离越小表示两个特征分布越接近。

FVD 是分布级指标，依赖特征提取器、样本数、视频长度、预处理和随机种子。它可能奖励生成常见的平均运动，却忽略少数高价值 prompt 的准确性；它也不能单独回答“这段视频是否保留了输入人物”。报告 FVD 时必须同时写清特征网络、采样设置和数据切分。

### 8.17.2 文本遵循与组合关系

视频文本评估至少要拆成：

- 对象是否出现；
- 属性是否正确；
- 数量是否正确；
- 空间关系是否正确；
- 动作是否发生；
- 时间顺序是否正确；
- 镜头描述是否遵循。

一个通用的视频-文本 embedding 分数可以反映整体相似度，但不适合替代关系和顺序测试。对于“红球在蓝球左边并绕过它”这样的 prompt，应使用对象跟踪和事件判定，而不是只测整段视频与文本的一个相似度。

### 8.17.3 VBench 的维度化思路

[VBench](https://arxiv.org/abs/2311.17982) 把视频生成质量拆成多个相对独立的维度，例如主体身份一致性、运动平滑度、时间闪烁和空间关系，并使用专门的 prompt 与评估方法。它的启发是把“视频质量”拆成可诊断的子问题，而不是只排一个总榜。

使用 benchmark 时仍需注意：维度和数据集是公开实验的定义，不一定覆盖业务任务。商业广告、教育演示、机器人仿真和电影镜头的权重不同，应该在公开 benchmark 之外加入业务切片。

### 8.17.4 人工评估和统计不确定性

人工偏好可以捕捉自动指标难以表达的自然度和连贯性，但也会受到 prompt、展示顺序、视频长度、屏幕和标注者经验影响。应记录：

- 评分标准和示例；
- 标注者数量与训练；
- pairwise 还是独立打分；
- 失败原因标签；
- 置信区间或标注一致性。

如果两个版本的偏好差异很小，不应只因为总票数略高就宣称能力提升。长视频还需要按时间段统计错误出现位置，避免短开头掩盖末尾崩溃。

### 8.17.5 一个实际评估表

| 维度 | 样本问题 | 观测方式 | 失败后果 |
| --- | --- | --- | --- |
| 空间画质 | 细节和文字是否清楚 | 人工、感知指标、OCR | 不可读或不可用 |
| 条件遵循 | 对象、属性、关系是否满足 | 检测、跟踪、事件判定 | 任务失败 |
| 身份保持 | 主体是否始终相同 | 跟踪、embedding、属性检查 | 角色和商品漂移 |
| 时间一致 | 是否闪烁和跳变 | 局部区域指标、人工 | 观看不适、编辑失败 |
| 运动自然 | 轨迹和姿态是否合理 | 关键点、光流、人工 | 动作不可信 |
| 物理与因果 | 接触和干预后果是否合理 | 规则、仿真、反事实 | 控制风险 |
| 安全与来源 | 是否滥用身份或伪造证据 | 审核、provenance、人工 | 合规和社会风险 |
| 成本 | 延迟、显存、单位时长成本 | 线上日志和硬件实测 | 无法部署 |

## 8.18 视频生成和机器人、自动驾驶的关系

### 8.18.1 共同点

视频生成和具身系统都需要建模未来：车辆下一秒的位置、行人是否穿越、机械臂动作是否导致物体移动。二者都受时空表示、遮挡、长程记忆和不确定性影响。

视频模型可以为具身系统提供预训练的视觉动态表示、数据增强或候选未来；world model 可以用生成的观察帮助规划器比较不同动作序列。

### 8.18.2 关键差异

控制系统必须面对真实动作的后果和安全约束。它至少需要：

1. 状态估计和传感器同步；
2. 可执行动作接口；
3. 目标、奖励或代价函数；
4. 碰撞和安全边界；
5. 实时反馈和故障处理；
6. 真实环境或高保真仿真验证。

视频生成器通常只需输出一条视觉序列，不必保证动作可执行，也不必提供对真实环境的反事实响应。因此不能因为一个视频模型能生成“汽车避开行人”的画面，就把它当成自动驾驶控制器。

### 8.18.3 从开放环样片到闭环系统

一个可操作的迁移路径是：

1. 用真实视频检查短期状态预测。
2. 加入动作条件，检查不同动作是否得到不同后果。
3. 在仿真器中做多步 rollout 和规划。
4. 用分布外场景和故障传感器测试不确定性。
5. 以极低风险和人工监督方式进入真实环境。

每一步都应保存失败轨迹，而不是只保留成功演示。安全性来自对失败的覆盖，不来自少量漂亮样片。

## 8.19 安全、版权与生成内容来源

### 8.19.1 视频风险为什么更难处理

视频同时包含图像、动作、声音、身份和时间因果，误导性更强。风险包括：

- 未经同意的肖像或声音模仿；
- 伪造新闻、监控和证据；
- 暴力、违法或危险行为的生成；
- 训练数据中的隐私和版权问题；
- 生成视频与真实记录混淆；
- 长视频中局部高风险内容逃过抽检。

### 8.19.2 纵深防护

安全系统应覆盖输入、生成过程、输出和发布环节：

1. 输入 prompt、参考图和身份授权检查。
2. 生成过程中对高风险条件和中间结果采样检查。
3. 输出视频做画面、音频、文字和身份风险审核。
4. 通过 provenance 或内容凭证记录生成来源和编辑历史。
5. 对高风险人物、真实事件和证据场景设置更严格的人工复核。
6. 记录模型版本、输入、seed、编辑操作和发布人。

水印或 provenance 有助于追踪来源，但不能替代内容真实性判断。水印可能被裁剪、转码或重新录制，来源记录也不能证明视频中的事件真实发生。

### 8.19.3 安全评估也要测时间维度

不能只审核第一帧。应测试：

- 风险对象是否在后续帧才出现；
- 身份和声音是否在中途切换；
- 局部编辑是否绕过整体审核；
- 拼接窗口是否丢失安全标签；
- 音画是否在后期错配造成误导。

对长视频，抽样策略、关键帧密度和事件级扫描都应成为系统配置的一部分。

## 8.20 可运行的 token、时序与 rollout 审计

下面的 demo 不生成视频，也不声称能测真实物理。它只用标准库计算三类可复核量：时空 token 和 latent 的算术成本、简化的时序质量指标，以及一个线性状态转移模型的 rollout 误差。frames 中的数值是人工构造的教学轨迹，不能当作模型评测结果。

代码中 `config["frames"]` 表示完整视频用于成本估算的帧数，`frames` 列表表示一段较短的人工指标窗口。二者有意分开：真实系统可以对完整视频分窗口计算身份、亮度和运动指标，但不能把一个 5 帧窗口的合格结果写成 16 帧视频整体合格。由于本 demo 还计算身份变化和二阶运动差分，指标窗口至少需要 3 帧；报告会显式输出窗口长度，并检查它没有超过完整视频长度。

~~~python
from math import isfinite, sqrt


def ceil_div(a, b):
    if not isinstance(a, int) or not isinstance(b, int) or a <= 0 or b <= 0:
        raise ValueError("ceil_div inputs must be positive integers")
    return (a + b - 1) // b


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    if not all(isfinite(value) for value in a + b):
        raise ValueError("vectors must contain finite values")
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    if not a:
        raise ValueError("vector must be non-empty")
    return sqrt(dot(a, a))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    if denominator == 0.0:
        raise ValueError("cosine is undefined for a zero vector")
    return dot(a, b) / denominator


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def audit_video(config, frames):
    for field in (
        "batch",
        "channels",
        "frames",
        "height",
        "width",
        "spatial_patch",
        "temporal_patch",
        "temporal_scale",
        "spatial_scale",
        "latent_channels",
        "steps",
        "prompt_tokens",
        "token_budget",
    ):
        if not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if config["guidance_scale"] < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if len(frames) < 3 or len(frames) > config["frames"]:
        raise ValueError("metric window must contain 3..config['frames'] frames")
    if not isfinite(config["guidance_scale"]):
        raise ValueError("guidance_scale must be finite")
    for frame in frames:
        if not isinstance(frame["center"], (int, float)) or not isfinite(frame["center"]):
            raise ValueError("frame center must be numeric")
        if not frame["identity"]:
            raise ValueError("identity vector must be non-empty")
        if any(
            not isinstance(value, (int, float)) or not isfinite(value)
            for value in frame["identity"]
        ):
            raise ValueError("identity vector must be numeric")
        if (
            not isinstance(frame["visible"], (int, float))
            or not isfinite(frame["visible"])
            or not 0.0 <= frame["visible"] <= 1.0
        ):
            raise ValueError("visibility must be in [0, 1]")
    raw_shape = (
        config["batch"],
        config["channels"],
        config["frames"],
        config["height"],
        config["width"],
    )
    spatial_tokens_per_frame = (
        ceil_div(config["height"], config["spatial_patch"])
        * ceil_div(config["width"], config["spatial_patch"])
    )
    framewise_tokens = config["frames"] * spatial_tokens_per_frame
    st_grid = (
        ceil_div(config["frames"], config["temporal_patch"]),
        ceil_div(config["height"], config["spatial_patch"]),
        ceil_div(config["width"], config["spatial_patch"]),
    )
    st_tokens = st_grid[0] * st_grid[1] * st_grid[2]

    latent_shape = (
        config["batch"],
        config["latent_channels"],
        ceil_div(config["frames"], config["temporal_scale"]),
        ceil_div(config["height"], config["spatial_scale"]),
        ceil_div(config["width"], config["spatial_scale"]),
    )
    pixel_elements = 1
    for dimension in raw_shape:
        pixel_elements *= dimension
    latent_elements = 1
    for dimension in latent_shape:
        latent_elements *= dimension
    latent_tokens = latent_shape[2] * latent_shape[3] * latent_shape[4]

    denoiser_calls = config["steps"] * (
        2 if config["guidance_scale"] > 1.0 else 1
    )
    cross_attention_cells = (
        config["steps"]
        * latent_tokens
        * config["prompt_tokens"]
        * (2 if config["guidance_scale"] > 1.0 else 1)
    )

    centers = [item["center"] for item in frames]
    second_diffs = [
        centers[i + 1] - 2 * centers[i] + centers[i - 1]
        for i in range(1, len(centers) - 1)
    ]
    identity_cosines = [
        cosine(frames[i - 1]["identity"], frames[i]["identity"])
        for i in range(1, len(frames))
    ]
    brightness_diffs = [
        abs(frames[i]["brightness"] - frames[i - 1]["brightness"])
        for i in range(1, len(frames))
    ]
    object_visibility = mean(item["visible"] for item in frames)

    observed_velocity = frames[1]["center"] - frames[0]["center"]
    rollout = [
        round(frames[0]["center"] + i * observed_velocity, 3)
        for i in range(len(frames))
    ]
    rollout_mae = mean(
        abs(predicted - observed)
        for predicted, observed in zip(rollout, centers)
    )

    report = {
        "raw_shape": raw_shape,
        "framewise_tokens": framewise_tokens,
        "spatiotemporal_grid": st_grid,
        "spatiotemporal_tokens": st_tokens,
        "token_ratio": round(st_tokens / framewise_tokens, 3),
        "latent_shape": latent_shape,
        "latent_element_ratio": round(latent_elements / pixel_elements, 6),
        "latent_tokens": latent_tokens,
        "denoiser_calls": denoiser_calls,
        "cross_attention_cells": cross_attention_cells,
        "motion_second_difference": round(mean(abs(v) for v in second_diffs), 4),
        "minimum_identity_cosine": round(min(identity_cosines), 4),
        "average_brightness_change": round(mean(brightness_diffs), 4),
        "object_visibility": round(object_visibility, 3),
        "metric_window_frames": len(frames),
        "linear_rollout": rollout,
        "linear_rollout_mae": round(rollout_mae, 4),
    }
    checks = {
        "tokens_within_budget": st_tokens <= config["token_budget"],
        "latent_ratio_under_demo_limit": report["latent_element_ratio"] < 0.01,
        "cfg_call_count_matches": denoiser_calls == config["steps"] * 2,
        "metric_window_within_video": len(frames) <= config["frames"],
        "identity_above_demo_threshold": report["minimum_identity_cosine"] > 0.98,
        "brightness_change_under_demo_limit": report["average_brightness_change"] < 0.04,
        "motion_change_under_demo_limit": report["motion_second_difference"] < 0.03,
        "object_visible_in_demo": object_visibility == 1.0,
        "linear_rollout_error_under_demo_limit": report["linear_rollout_mae"] < 0.02,
    }
    report["checks"] = checks
    report["audit_consistent"] = all(checks.values())
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

report = audit_video(config, frames)
for key, value in report.items():
    print(f"{key}={value}")


def expects_value_error(function):
    try:
        function()
    except ValueError:
        return True
    return False


boundary_checks = {
    "zero_patch_rejected": expects_value_error(
        lambda: audit_video({**config, "spatial_patch": 0}, frames)
    ),
    "too_short_metric_window_rejected": expects_value_error(
        lambda: audit_video(config, [])
    ),
    "oversized_metric_window_rejected": expects_value_error(
        lambda: audit_video(
            {**config, "frames": 2},
            frames,
        )
    ),
    "zero_identity_rejected": expects_value_error(
        lambda: audit_video(
            config,
            [{**frames[0], "identity": [0.0, 0.0]}],
        )
    ),
    "mismatched_identity_rejected": expects_value_error(
        lambda: audit_video(
            config,
            [{**frames[0], "identity": [1.0]}, frames[1]],
        )
    ),
    "invalid_visibility_rejected": expects_value_error(
        lambda: audit_video(
            config,
            [{**frames[0], "visible": 2.0}],
        )
    ),
}
assert all(report["checks"].values())
assert all(boundary_checks.values())
print(f"boundary_checks={boundary_checks}")
print(
    "all_checks_passed="
    f"{report['audit_consistent'] and all(boundary_checks.values())}"
)
~~~

运行这段代码得到的关键输出为：

~~~text
raw_shape=(1, 3, 16, 256, 256)
framewise_tokens=4096
spatiotemporal_grid=(8, 16, 16)
spatiotemporal_tokens=2048
token_ratio=0.5
latent_shape=(1, 4, 4, 32, 32)
latent_element_ratio=0.005208
latent_tokens=4096
denoiser_calls=48
cross_attention_cells=2752512
motion_second_difference=0.0133
minimum_identity_cosine=0.9987
average_brightness_change=0.015
object_visibility=1.0
metric_window_frames=5
linear_rollout=[0.1, 0.21, 0.32, 0.43, 0.54]
linear_rollout_mae=0.002
checks={'tokens_within_budget': True, 'latent_ratio_under_demo_limit': True, 'cfg_call_count_matches': True, 'metric_window_within_video': True, 'identity_above_demo_threshold': True, 'brightness_change_under_demo_limit': True, 'motion_change_under_demo_limit': True, 'object_visible_in_demo': True, 'linear_rollout_error_under_demo_limit': True}
boundary_checks={'zero_patch_rejected': True, 'too_short_metric_window_rejected': True, 'oversized_metric_window_rejected': True, 'zero_identity_rejected': True, 'mismatched_identity_rejected': True, 'invalid_visibility_rejected': True}
audit_consistent=True
all_checks_passed=True
~~~

这些数字能验证以下算术：

1. 逐帧切分得到 4096 个 token，时间 patch 为 2 时得到 2048 个 token。
2. 4 个 latent 通道、时间压缩 4、空间压缩 8 时，理想化元素比例为 0.005208。
3. 24 个采样步和 CFG 对应 48 次 denoiser 调用。
4. 人工轨迹的最小 identity cosine、亮度变化和二阶差分满足预先写明的教学阈值。
5. 线性 rollout 的 MAE 很小，只说明这组人工轨迹接近匀速运动，不说明任何真实模型已经学会世界动力学。

若把其中一帧的 visible 改为 False，或把某个 identity 向量改得差异很大，报告会指出对应的失败项。实际系统不应直接沿用这些阈值；阈值需要根据分辨率、主体类型、摄像机运动和任务损失从验证集估计。

## 8.21 从失败视频反推系统问题

### 8.21.1 先定位失败发生在哪一层

看到一个坏视频时，应该按数据流逐层问：

1. 输入视频或条件是否已错位、裁剪错误或损坏？
2. tokenizer 或 VAE 重建是否已经丢失文字、细节或时间关系？
3. 条件编码是否被截断、错配或注入到错误的时间位置？
4. denoiser 是否只在局部窗口建模，导致长期漂移？
5. scheduler、prediction type、CFG 或精度是否配置错误？
6. 视频解码、插帧、超分辨率或拼接是否引入了伪影？
7. 评估器是否把真实运动误判为闪烁，或漏掉了局部错误？

这条路径比笼统地说“模型不够大”更有诊断价值。

### 8.21.2 用最小对照实验分离原因

可以设计下面几组对照：

- 固定 seed，只改变 temporal module，观察身份和运动是否变化；
- 固定模型，只改变 VAE，比较重建和生成质量；
- 固定 prompt，只改变视频长度，观察错误从哪一帧开始；
- 固定长度，只改变 fps，检查模型是否依赖固定时间尺度；
- 固定文本，只移除姿态或深度条件，测控制增益和副作用；
- 固定输出，只关掉 CFG，检查过强条件是否造成动作僵硬；
- 固定模型和输入，分别评估开放环与闭环 rollout。

每次只改变一个变量，才能把现象和原因联系起来。记录每个实验的模型版本、VAE、scheduler、采样步数、seed、硬件和后处理设置。

### 8.21.3 失败样本应按类型留存

建议给失败视频加结构化标签：flicker、identity_drift、motion_jump、geometry_error、occlusion_error、prompt_relation_error、physics_error、safety_error 和 audio_sync_error。标签不是模型能力本身，而是为了观察版本迭代后哪类错误减少、哪类错误被牺牲。

## 8.22 独立练习

1. 一段 6 秒、30 fps、512 x 512 的视频使用 16 x 16 空间 patch，分别计算逐帧 token 数和时间 patch 为 2 时的 token 数。
2. 解释为什么时间 patch 减少 token 后，快速运动和小目标可能更难生成。
3. 给定 C_z=4、f_t=4、f_s=8，推导 latent 元素量相对于 RGB 视频的比例，并说明它不等于显存比例的原因。
4. 比较一次性全时空生成、关键帧加插帧和滑动窗口生成三种路线的长视频误差来源。
5. 为“一个红球被挡住后从桌子另一侧出现”设计身份保持、物体持久性和遮挡恢复三个指标。
6. 说明 brightness difference 为什么不能单独作为闪烁指标，并提出一个背景区域对照实验。
7. 设计一个能区分“主体移动”和“摄像机推进”的视频条件测试。
8. 写出一个动作条件 world model 的状态转移方程，并解释状态 s_t、动作 a_t、观察 o_t 的区别。
9. 说明一步预测准确为什么不代表多步闭环 rollout 准确，并设计一个开放环/闭环对照。
10. 比较像素重建、latent dynamics 和 JEPA 类预测型表征在任务相关信息、可解释性和生成画质方面的取舍。
11. 为“机器人把杯子推到桌边”设计一个反事实测试，至少改变两个环境或动作条件。
12. 解释 FVD 的均值和协方差分别比较什么，以及为什么报告 FVD 时必须记录特征提取器和预处理。
13. 用一个表格设计视频生成评估，至少包含画质、条件遵循、身份、运动、物理、安全和单位时长成本。
14. 运行本章 demo，将一帧的 identity、brightness 或 visible 改坏，说明报告中的哪些量发生变化。
15. 选择一个长视频失败案例，按“输入、压缩、条件、denoiser、采样、解码、评估”顺序提出最小对照实验。

## 8.23 本章小结

视频生成不是把图片重复播放，而是在空间信号之外建模时间、运动、身份和事件关系。视频张量的帧数、分辨率和 fps 直接决定 token 与显存成本；时空 patch 和视频 latent 可以降低成本，却会以细节、快速动作或短暂事件的损失为代价。

Video diffusion 在视频 latent 上进行加噪和去噪，级联模型、全时空模型、离散视频 token、自回归 Transformer 和 DiT 分别代表不同的状态空间与计算组织方式。文生视频适合表达高层意图，姿态、深度、光流、摄像机轨迹和音频等条件用于补充精细控制，但多条件也会带来时间对齐和冲突处理问题。

时序一致性必须拆成闪烁、身份、纹理、几何、运动和物体持久性来测；物理一致性还要进一步检查接触、碰撞、遮挡和反事实后果。FVD、视频文本相似度和 VBench 等工具各有覆盖范围，不能用一个总分替代任务型和人工评估。

World model 的核心是状态、观察、动作和状态转移。视频生成模型可能包含世界动态的部分统计规律，但只有当模型能在动作条件下稳定预测、支持 rollout 和规划，并在任务和安全约束上验证，才有理由把它称为控制意义上的 world model。下一章将转向语音与音频生成，继续追踪连续信号如何被压缩、建模和接入多模态系统。

### 资料与阅读顺序

1. [Video Diffusion Models](https://arxiv.org/abs/2204.03458)：视频扩散、图像和视频联合训练、空间与时间延展。
2. [Imagen Video](https://arxiv.org/abs/2210.02303)：级联视频扩散、空间/时间超分辨率、v-parameterization 和 progressive distillation。
3. [Make-A-Video](https://arxiv.org/abs/2209.14792)：从文本-图像知识和无监督视频学习文生视频。
4. [Phenaki](https://arxiv.org/abs/2210.02399)：离散视频 token、变量长度和时间变化文本条件。
5. [Lumiere](https://arxiv.org/abs/2401.12945)：Space-Time U-Net 和整段时间生成的公开设计。
6. [VideoPoet](https://arxiv.org/abs/2312.14125)：多模态 decoder-only Transformer 的视频生成路线。
7. [CogVideoX](https://arxiv.org/abs/2408.06072)：3D VAE、Diffusion Transformer 和视频 token 压缩的公开案例。
8. [World Models](https://arxiv.org/abs/1803.10122)：压缩环境表示、动态模型与 imagined rollout 的早期工作。
9. [DreamerV3](https://arxiv.org/abs/2301.04104)：在学习到的世界模型中想象未来并改进策略。
10. [V-JEPA 2](https://arxiv.org/abs/2506.09985)：自监督视频表征、动作条件 latent world model 和规划实验。
11. [FVD](https://arxiv.org/abs/1812.01717)：视频生成的 Fréchet Video Distance 与评估挑战。
12. [VBench](https://arxiv.org/abs/2311.17982)：把视频生成质量拆成多个维度的 benchmark。
13. [Sora 官方技术说明](https://openai.com/index/sora/)：公开的 patch 化视频表示和产品层能力说明；未公开内容不应从页面之外推断。
