# 第七章：Stable Diffusion 与 DALL·E——把扩散公式组织成文生图系统

上一章讲的是 diffusion 的数学骨架：如何加噪，如何预测噪声或 velocity，如何用 scheduler 从噪声走回数据。本章把这套骨架放进真实的文生图 pipeline，回答“一个 prompt 到底经过了哪些模块，哪些模块承担什么责任”。

Stable Diffusion 和 DALL·E 经常被放在同一个句子里，但它们不是同一条工程路线：

- Stable Diffusion 代表一类在 VAE latent 空间进行扩散的系统；
- 早期 DALL·E 把图像离散化为 image tokens，再用自回归 Transformer 生成；
- DALL·E 2 使用文本到 CLIP image embedding 的 prior，再由扩散 decoder 生成图像；
- 后续文生图系统还可能使用 DiT、不同的文本编码器、flow-like 路径、蒸馏和多种控制模块。

本章不根据产品名称猜内部结构，而是把可公开核对的模块、公式和接口分开讲。读者会从完整的系统责任链理解：

1. VAE 为什么降低空间成本，却会带来重建边界；
2. text encoder 为什么不是“把 prompt 变成一个向量”这么简单；
3. denoiser、cross-attention、scheduler 和 CFG 如何共同决定一条采样轨迹；
4. negative prompt、seed、image-to-image、inpainting、outpainting 和 ControlNet 分别控制什么；
5. LoRA 个性化为什么既便宜又容易过拟合；
6. DALL·E 的离散 token、CLIP latent 和扩散 decoder 如何与 latent diffusion 比较；
7. 质量、文本遵循、组合关系、文字可读性、安全和成本应该怎样分别评估。

## 0. 研究对象、资料与边界

### 0.1 本章研究的是系统契约

一个文生图系统不只是一个 denoiser。它至少包含：

~~~text
prompt
    -> tokenizer / text encoder
    -> positive and optional negative condition
    -> initial noise or encoded image latent
    -> denoiser + scheduler + optional guidance/control
    -> final latent
    -> VAE decoder
    -> image and system-level checks
~~~

每个箭头都是契约：

- tokenizer 决定文本如何切分和截断；
- text encoder 决定条件表示；
- denoiser 决定如何利用状态、时间和条件；
- scheduler 决定实际更新轨迹；
- VAE 决定像素和 latent 的转换；
- control 分支决定结构约束如何进入主干；
- 后处理和安全模块决定输出如何交付。

如果只更换其中一个模块，却没有同步检查 scaling、dtype、shape、时间网格和 prediction type，系统可能仍然运行，但生成质量已经不可比较。

### 0.2 资料层级

| 资料类型 | 能支持的内容 | 不能直接支持的内容 |
| --- | --- | --- |
| Latent Diffusion 原始论文 | latent diffusion、VAE、条件 cross-attention 的研究路线 | 任意 Stable Diffusion 版本的全部实现 |
| DALL·E 与 DALL·E 2 论文 | 公开论文中的离散 token、CLIP latent prior 和 decoder | 后续闭源产品的私有结构 |
| ControlNet 原始论文 | 结构条件分支的设计和实验 | 任意控制模型的兼容性 |
| Diffusers 官方文档 | 某版本 pipeline、scheduler 和参数 API | API 默认值就是最佳质量 |
| 开源 Stable Diffusion 仓库 | 特定 commit 的代码和配置 | 当前所有社区 checkpoint 的行为 |
| T2I-CompBench、GenEval、CLIPScore 等论文 | 公开评估任务和指标背景 | 一个指标等于人类整体偏好 |
| 本章 demo | shape、token、调用次数和预算算术 | 真实图像质量、版权安全和 prompt 理解 |

产品名称只能作为检索入口，不能作为架构证据。若一个系统没有公开其 text encoder、denoiser、VAE 或 scheduler，书稿只能说明公开可验证的部分，并把其余部分标成未知。

### 0.3 初学者和专家的两种地图

初学者可以把 Stable Diffusion 想成四步：

1. 把 prompt 变成条件；
2. 从随机 latent 开始；
3. 反复去噪；
4. 把 latent 解码成图片。

专家需要进一步记录：

- 图像归一化范围；
- VAE latent 的通道和 scaling；
- 文本最大长度和截断方式；
- 正向、负向条件是否共享 encoder；
- CFG 每一步需要几次 denoiser forward；
- scheduler 接受 epsilon、x0 还是 v；
- img2img 的 strength 如何映射到时间步；
- ControlNet 条件是在像素空间还是 latent 对齐；
- checkpoint、精度、seed 和硬件版本。

### 0.4 本章不做的承诺

本章不会声称某个产品在所有 prompt、语言、分辨率和安全场景下都可靠，也不会把公开论文的实验分数写成当前服务的保证。文生图的质量、版权、身份模仿和高风险内容处理都需要目标系统的独立评估和治理。

## 1. 文生图系统到底要完成什么

### 1.1 条件分布与系统输出

给定 prompt c，文生图系统希望从条件分布采样：

~~~math
x\sim p_\theta(x\mid c).
~~~

x 是最终图片，c 不一定是原始字符串，而是经过 tokenizer 和 text encoder 后的条件表示。若还有结构控制 C、输入图片 x_in 或 mask m，可以写成：

~~~math
x\sim p_\theta(x\mid c,C,x_{\mathrm{in}},m).
~~~

不同任务只是条件集合不同：

- text-to-image：只有文本条件；
- image-to-image：文本加输入图像 latent；
- inpainting：文本加输入图像和 mask；
- ControlNet：文本加边缘、深度或姿态等结构条件；
- 个性化生成：文本加 checkpoint、LoRA 或 subject embedding。

### 1.2 生成成功不是“画面好看”一个事件

一个业务任务可能同时要求：

- prompt 中的对象出现；
- 对象属性正确；
- 对象数量正确；
- 空间关系正确；
- 文字可读；
- 风格符合要求；
- 多次采样有足够多样性；
- 不触发不允许的内容策略。

因此成功事件不应只写成“图片看起来不错”。例如商品海报要求文字正确时，审美高分不能抵消标题乱码；姿态控制要求骨架一致时，整体相似度不能替代关节关系。

### 1.3 Stable Diffusion 只是一个系统家族名称

“Stable Diffusion”在不同版本、社区 checkpoint 和 pipeline 中可能对应不同：

- VAE；
- text encoder；
- U-Net 或 Transformer denoiser；
- scheduler；
- control adapter；
- LoRA；
- tokenizer 和最大文本长度；
- 分辨率和 latent scaling。

因此本章使用“Stable Diffusion 类系统”描述公开的 latent diffusion pipeline，涉及具体参数时会说明那是示例配置，不把它当作所有版本的固定事实。

## 2. Stable Diffusion 类系统的模块责任

### 2.1 总体数据流

训练时可以抽象为：

~~~text
image x0
    -> VAE encoder
    -> clean latent z0
    -> add noise at timestep t
    -> noisy latent zt
    -> denoiser(zt, t, text condition)
    -> epsilon / x0 / v target
    -> loss
~~~

推理时则是：

~~~text
prompt
    -> tokenizer
    -> text encoder
    -> condition embeddings
random latent zT
    -> denoiser + scheduler repeated K times
clean latent z0_hat
    -> VAE decoder
image
~~~

VAE、text encoder、denoiser 和 scheduler 不是互相替代的组件。VAE 不负责理解 prompt，text encoder 不负责恢复像素，scheduler 不负责学习语义，denoiser 也不能绕过 VAE 的信息损失。

### 2.2 VAE encoder 和 decoder

设输入图片为 x，VAE encoder 和 decoder 分别为 E_phi、D_psi：

~~~math
z=E_\phi(x),
\qquad
\hat{x}=D_\psi(z).
~~~

如果 batch 为 B，VAE scale factor 为 f，latent channel 为 C_z，输入尺寸为 H×W，常见形状近似为：

~~~math
z\in
\mathbb{R}^{B\times C_z\times H/f\times W/f}.
~~~

H 和 W 如果不能整除 f，实际实现可能先 resize、pad 或使用卷积边界规则，不能只用整数除法猜最终 shape。

VAE 的责任是把像素表示转换成便于扩散的连续状态，并尽量保留生成任务需要的视觉信息。它不是无损压缩：

- 高频纹理可能被平滑；
- 小字和细线可能丢失；
- 颜色范围和归一化会影响 decoder；
- decoder 会引入重建误差；
- 换 VAE 可能改变 denoiser 看到的 latent 分布。

### 2.3 latent scaling 是接口的一部分

很多 latent diffusion 实现会在送入 denoiser 前使用缩放因子 s_z：

~~~math
z_{\mathrm{model}}=s_z z_{\mathrm{vae}}.
~~~

解码前需要使用匹配的逆变换：

~~~math
\hat{x}
=
D_\psi
\left(
\frac{z_{\mathrm{model}}}{s_z}
\right).
~~~

这里的 s_z 不是可以随意省略的装饰。如果训练时 denoiser 看到的 latent 方差与推理时不同，采样轨迹会偏离训练分布，可能表现为黑图、过曝、细节异常或 prompt 遵循下降。

检查 checkpoint 时要把 VAE、scaling factor、像素归一化范围、dtype 和 decoder 后处理一起记录。

### 2.4 Text encoder 的责任

prompt 先经过 tokenizer，再经过 text encoder：

~~~text
prompt
    -> tokenizer
    -> text token ids
    -> text encoder
    -> text hidden states
~~~

denoiser 可能接收整个 text hidden sequence，而不是一个简单的平均向量。text encoder 的版本会影响：

- 词语和短语的语义；
- 长 prompt 的截断；
- 多语言和特殊符号；
- 属性、关系和风格的表达；
- cross-attention 的 key/value 形状。

若 prompt 超过最大长度，后半部分可能被截断；如果 tokenizer 与 text encoder 不匹配，可能从 shape 上看不出错误，却导致语义变化。训练和推理必须固定 tokenizer、text encoder、模板和最大长度。

### 2.5 Denoiser 的责任

denoiser 接收 noisy latent、时间步和条件，输出训练约定的参数化：

~~~math
\hat{\epsilon}
=
\epsilon_\theta(z_t,t,E_c),
~~~

或：

~~~math
\hat{v}
=
v_\theta(z_t,t,E_c).
~~~

它负责估计当前噪声状态下的去噪方向，不负责决定 scheduler 的时间网格，也不负责把字符串直接变成像素。输出通道、归一化、时间 embedding 和条件路径必须与 scheduler 配套。

### 2.6 Scheduler 的责任

scheduler 负责把 denoiser 输出变成下一状态。它可能管理：

- alpha、sigma 和时间步；
- prediction type；
- 当前和下一时间步；
- 随机方差；
- solver 更新；
- clipping 和 scaling。

相同 denoiser 换 scheduler，可能得到不同结果；相同 scheduler 换 prediction type，也可能完全不兼容。实验记录必须把 scheduler 的配置保存下来，而不能只写“用了 Euler”。

## 3. Latent 空间的成本与形状

### 3.1 空间 cell 的压缩比

像素空间的空间位置数量为 H W，若 VAE 在高宽方向都压缩 f 倍，latent 空间位置数量约为：

~~~math
R_{\mathrm{spatial}}
=
\frac{(H/f)(W/f)}{HW}
=
\frac{1}{f^2}.
~~~

若同时考虑 RGB 通道和 latent channel：

~~~math
R_{\mathrm{element}}
=
\frac{C_z(H/f)(W/f)}
{3HW}
=
\frac{C_z}{3f^2}.
~~~

例如 f=8、C_z=4：

~~~math
R_{\mathrm{element}}
=
\frac{4}{3\times64}
=
\frac{1}{48}
\approx0.0208.
~~~

这只是状态元素数量的粗略比例。实际成本还包括 VAE、denoiser 宽度、attention、activation、精度、CFG forward 次数和采样步数。

### 3.2 具体形状例子

以 B=1、H=W=512、f=8、C_z=4 为例：

~~~math
z\in\mathbb{R}^{1\times4\times64\times64}.
~~~

latent 空间位置为 4096，而 RGB 像素位置为 262144。若把 latent 展平成 DiT patch token，patch size P=2，则 token 数是：

~~~math
N_z
=
\frac{64}{2}\times\frac{64}{2}
=
1024.
~~~

如果 P=1，则 token 数为 4096，attention 成本可能明显上升。latent 压缩和 patch 化是两个不同层次的降维，不能只记录其中一个。

### 3.3 分辨率不整除的处理

若 H/f 或 W/f 不是整数，系统可能：

- resize 到可整除尺寸；
- pad 后编码；
- crop 掉边界；
- 使用卷积输出的 floor/ceil 规则；
- 在 decoder 后再裁回目标尺寸。

输入 513×512 和 512×512 不一定只差一个像素的成本，它们可能进入不同的 latent shape。生产 pipeline 要把原始尺寸、预处理尺寸、latent 尺寸和最终裁剪记录在一起。

## 4. Text condition 与 cross-attention

### 4.1 文本 token 如何进入 denoiser

设 latent hidden states 为 H_z，文本 hidden states 为 E_c：

~~~math
Q=H_z W_Q,
\qquad
K=E_c W_K,
\qquad
V=E_c W_V.
~~~

cross-attention 为：

~~~math
\operatorname{Attn}(H_z,E_c)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V.
~~~

latent token 数为 N_z，文本 token 数为 N_c，单层注意力 score 的 cell 数近似为：

~~~math
C_{\mathrm{cross}}
\propto
N_zN_c.
~~~

如果每个采样步有 L 个 cross-attention 层、采样步数为 K，则粗略交互量为：

~~~math
C_{\mathrm{total}}
\propto
K L N_zN_c.
~~~

这解释了为什么高分辨率、长 prompt、多层 cross-attention 和多步采样会共同推高成本。

### 4.2 cross-attention 不是词到像素的硬对齐

“模型在生成汽车区域时 attend 到 car”是有用的直觉，但不是严格的可解释保证。attention 权重会随层、时间步、seed 和采样器变化；一个词可能参与多个区域，多个词也可能共同影响同一区域。

如果业务要求对象框、文字区域或结构约束，应使用显式 grounding、ControlNet、mask 或后验检测，而不能只把 attention heatmap 当作证据。

### 4.3 长 prompt、截断和权重

prompt 经过 tokenizer 后会有最大长度。超出部分可能被截断，也可能被分段编码，具体行为取决于实现。某些工具还提供 token weighting 或 prompt embedding 拼接，但这不是所有 checkpoint 的通用语义。

工程上要记录：

- 原始 prompt；
- tokenized 长度；
- 截断位置；
- 分段或拼接方式；
- text encoder 版本；
- positive/negative 条件的实际长度。

如果 prompt 的关键约束在截断部分，生成失败不应首先归因于 denoiser。

## 5. Prompt、negative prompt 与 CFG

### 5.1 Prompt 是条件，不是程序

prompt 可以提供对象、属性、关系、风格和构图意图，例如：

~~~text
a watercolor illustration of a red bicycle beside a yellow house,
soft morning light, wide composition
~~~

但 prompt 通常不是严格执行的程序。模型可能：

- 漏掉一个属性；
- 把两个对象关系画错；
- 把数量变成常见先验；
- 在长 prompt 中忽略后半段；
- 用训练分布中的常见构图替代罕见组合。

因此 prompt 遵循必须用组合关系、计数、空间和属性评估，而不能只看单张样例。

### 5.2 Negative prompt 的真实边界

negative prompt 常被作为负条件 c_neg 输入 CFG。它更接近“沿着一个条件方向远离另一种条件”，而不是一个逻辑上的禁止集合。它可能帮助降低模糊、低质量或某类常见伪影，但不保证：

- 目标对象永不出现；
- 结构错误一定消失；
- 版权或安全风险被识别；
- 所有 checkpoint 都有相同响应。

negative prompt 的效果受 text encoder、训练数据、CFG scale、scheduler 和 denoiser 共同影响。

### 5.3 CFG 的公式和成本

设 positive condition 为 c_pos，negative 或空 condition 为 c_neg。若两次 denoiser 预测分别为 epsilon_pos 和 epsilon_neg，常见公式是：

~~~math
\hat{\epsilon}_{\mathrm{guided}}
=
\epsilon_{\mathrm{neg}}
+
s
\left(
\epsilon_{\mathrm{pos}}
-
\epsilon_{\mathrm{neg}}
\right).
~~~

这里 s 是 guidance scale。按这一约定：

- s=0 是负条件预测；
- s=1 等于正条件预测；
- s>1 放大从负条件到正条件的方向。

某些库的参数名和内部约定不同，不能只看变量名。CFG 通常使每个采样步至少需要正负两次 denoiser 计算；实现也可以把两种条件拼成一个 batch 来提高硬件利用率，但数学上的 forward 数仍应如实记录。

### 5.4 guidance scale 的折中

s 较低时，样本可能更有多样性，但 prompt 遵循较弱；s 较高时，条件方向更强，但可能出现：

- 过饱和；
- 局部结构被拉坏；
- 多个对象粘连；
- 纹理不自然；
- seed 间多样性下降。

比较 guidance 时应固定 prompt、seed、scheduler、步数、分辨率和 VAE，并同时报告条件遵循、图像质量、多样性和失败率。

## 6. Seed、scheduler 与实验可复现

### 6.1 seed 控制的是什么

seed 通常影响初始噪声和采样过程中的随机项：

~~~text
seed
    -> initial noise
    -> denoising trajectory
    -> generated image
~~~

在相同 checkpoint、VAE、text encoder、scheduler、精度、硬件和后处理下，相同 seed 往往能得到相同或接近的结果。它不是跨框架、跨版本的全局内容 id。

### 6.2 为什么相同 seed 仍可能改变

结果可能因为以下变化而不同：

- text encoder 或 tokenizer 版本；
- scheduler 时间网格；
- VAE scaling；
- GPU kernel 和精度；
- batch 中的并行顺序；
- CFG 拼 batch 的实现；
- 采样器随机项；
- image resize 和 crop；
- checkpoint 或 LoRA 权重。

因此复现实验要保存完整配置，而不是只保存 seed。

### 6.3 scheduler 比较的实验矩阵

至少固定：

- checkpoint；
- VAE；
- prompt 和 negative prompt；
- seed 集合；
- 分辨率；
- batch；
- precision；
- guidance scale；
- 输出后处理。

再改变 scheduler、步数或 solver。否则一个所谓的 scheduler 提升，可能其实来自不同的初始噪声或不同的 latent scaling。

## 7. Image-to-Image、Inpainting 与 Outpainting

### 7.1 Image-to-Image

Image-to-image 先把输入图片编码到 latent，再按某个噪声强度得到 z_t：

~~~math
z_t
=
\sqrt{\bar{\alpha}_t}z_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon.
~~~

然后从这个中间状态开始反向采样。denoising strength 通常决定 t 或时间步起点，但不同 pipeline 的参数映射可能不同。

强度低：

- 原图结构和颜色保留较多；
- prompt 主要做局部调整；
- 输入噪声较少。

强度高：

- 模型有更大自由度；
- 原图身份和布局更容易改变；
- 输出更接近从噪声重新生成。

不要把 strength 当成一个跨实现统一的物理量。应在目标 pipeline 中核对它如何转换为 scheduler timesteps。

### 7.2 Inpainting 的 mask 语义

设 m=1 表示需要重绘的区域，m=0 表示保留的区域，一个简化的混合状态可以写成：

~~~math
z_t^{\mathrm{mix}}
=
m\odot z_t^{\mathrm{new}}
+
(1-m)\odot z_t^{\mathrm{orig}}.
~~~

真实实现可能在像素空间、latent 空间或不同时间步混合，也可能对边界做额外处理。必须先确认 mask 的 0/1 语义，否则会把保留区域和重绘区域反过来。

Inpainting 的难点包括：

- mask 边缘接缝；
- 光照和透视一致；
- 新旧对象的语义冲突；
- 原图上下文是否被过度改动；
- mask 外区域是否发生漂移。

### 7.3 Outpainting

Outpainting 把画布扩大，在新增区域生成内容，同时保持原区域的结构和风格。它比普通 text-to-image 多一个边界一致性问题：

- 新旧区域的光照要衔接；
- 透视和地平线要连续；
- 原区域不能被无意重绘；
- prompt 的新内容不能破坏原主题。

评估 outpainting 时要分别看新增区域质量、边界连续性和原区域保持率。

## 8. ControlNet 与结构控制

### 8.1 ControlNet 解决什么问题

prompt 适合表达“画什么”，但很难精确表达一个姿态骨架、边缘轮廓或深度布局。ControlNet 类方法引入额外结构条件 C，例如：

- canny edge；
- depth；
- pose；
- segmentation；
- sketch；
- line art。

它的目标不是让文本条件消失，而是让主干同时接收语义和结构。

### 8.2 残差注入的抽象

设 base denoiser 第 l 层的 hidden state 为 h_l，控制分支输出残差 r_l(C)，可以抽象为：

~~~math
h_l^{\mathrm{out}}
=
h_l^{\mathrm{base}}
+
\lambda_l r_l(C).
~~~

lambda_l 是控制强度或层级系数。真实实现还包含零初始化、复制 block、卷积和多尺度特征等细节，不能用一个加法式子替代完整代码。

### 8.3 结构条件的预处理契约

ControlNet 的条件图不是任意图片。要记录：

- 条件生成器版本；
- 颜色和通道；
- 空间尺寸；
- 是否归一化；
- 边缘阈值、深度模型或姿态模型；
- 条件与原图是否对齐；
- control scale 和注入层。

边缘图错位一个 crop，模型可能仍然生成一张“合理图片”，但姿态和结构已经不再对应输入。

### 8.4 prompt 与 ControlNet 的责任分工

常见理解是：

- prompt 提供对象、属性和风格；
- ControlNet 提供位置、轮廓、深度或姿态；
- scheduler 和 guidance 决定采样路径；
- VAE 决定最终像素还原。

这是责任分工，不是严格隔离。prompt 仍会改变结构解释，control scale 也会影响语义表现。评估应分别做只改 prompt、只改控制图和同时改变两者的配对实验。

## 9. LoRA 与个性化生成

### 9.1 LoRA 的参数化

对一个冻结权重 W，LoRA 用低秩更新表示：

~~~math
W'
=
W+\frac{\alpha}{r}BA,
~~~

其中 A、B 是低秩矩阵，r 是 rank，alpha 是缩放系数。训练时只更新 A、B 或少量相关参数。

若原矩阵 $W\in\mathbb{R}^{d_{out}\times d_{in}}$，一个常见约定是 $A\in\mathbb{R}^{r\times d_{in}}$、$B\in\mathbb{R}^{d_{out}\times r}$，并要求 $1\leq r\leq\min(d_{in},d_{out})$。在这个约定下，LoRA 新增的可训练参数量为 $r(d_{in}+d_{out})$；alpha 只改变更新的缩放，不改变参数量。不同库可能交换 A/B 的命名，因此计算参数量时要以实际矩阵 shape 为准。

在 diffusion 系统中，LoRA 可能插入 text encoder、denoiser attention、projection 或其他模块。插入位置决定它更可能改变语言条件、风格、角色特征还是局部结构。

### 9.2 为什么个性化便宜

LoRA 不需要复制和更新全部 base model 参数，通常降低：

- 可训练参数量；
- 优化器状态；
- checkpoint 体积；
- 多风格部署成本。

但计算不一定按参数比例下降。训练仍要执行完整或大部分 denoiser forward，激活和 VAE 成本仍然存在。

### 9.3 过拟合和身份风险

个性化数据过少或重复过高时，LoRA 可能：

- 只记住训练构图；
- 把人物身份绑定到无关背景；
- 污染 base model 的通用风格；
- 对不同 prompt 泛化差；
- 放大隐私、肖像和版权风险。

评估至少要包含训练主体的不同角度、不同背景、不同构图、负面 prompt、未见 prompt 和不应出现的属性。能复现训练图不等于个性化成功。

## 10. DALL·E 路线与 Stable Diffusion 的比较

### 10.1 早期 DALL·E：离散 image token 自回归

早期 DALL·E 路线先用离散图像 tokenizer 把图片变成 image tokens，再让 Transformer 建模文本和图像 token：

~~~text
image
    -> discrete image tokenizer
    -> image token sequence
text + previous image tokens
    -> autoregressive Transformer
    -> next image token
~~~

若文本 token 为 u，图像 token 为 y_1 到 y_M，条件分布为：

~~~math
p(y_{1:M}\mid u)
=
\prod_{m=1}^{M}
p_\theta(y_m\mid u,y_{<m}).
~~~

训练目标是 next-token negative log likelihood：

~~~math
\mathcal{L}_{\mathrm{AR}}
=
-\sum_{m=1}^{M}
\log p_\theta(y_m\mid u,y_{<m}).
~~~

优点是与语言模型训练形式统一，文本和图像 token 可以放进同一自回归框架；代价是高分辨率会带来很长的图像 token 序列，生成延迟和 tokenizer 质量都很关键。

### 10.2 token 成本的手算

如果图像 tokenizer 输出 G×G 个 image tokens，图像部分长度为：

~~~math
M=G^2.
~~~

如果文本长度为 T_text，总序列长度近似为：

~~~math
T_{\mathrm{AR}}
=
T_{\mathrm{text}}+G^2+T_{\mathrm{special}}.
~~~

自回归生成还要逐 token 产生 M 次输出。KV cache 可以降低部分重复计算，但不能消除长序列的总生成步数。

### 10.3 DALL·E 2：CLIP image latent prior 与 decoder

DALL·E 2 论文描述的路线可以抽象为两阶段：

~~~text
text
    -> text embedding
    -> prior
    -> CLIP image embedding
    -> diffusion decoder
    -> image
~~~

用概率表示：

~~~math
z_{\mathrm{img}}
\sim
p_\theta(z_{\mathrm{img}}\mid z_{\mathrm{text}}),
~~~

~~~math
x
\sim
p_\psi(x\mid z_{\mathrm{img}},z_{\mathrm{text}}).
~~~

第一阶段 prior 预测图像语义 embedding，第二阶段 decoder 根据图像 embedding 和相关条件生成图片。它和 Stable Diffusion 的主要区别不只是“一个用了 CLIP、一个用了 VAE”，而是条件在系统中的位置不同：

- Stable Diffusion 类系统通常直接用 text encoder hidden states 条件化 latent denoiser；
- DALL·E 2 的公开论文路线先预测 CLIP image representation，再由 decoder 生成图片。

这两种路线都可能使用 diffusion，但 diffusion 所处的阶段和条件接口不同。

### 10.4 产品名称不能替代公开结构

对于后续 DALL·E 产品或闭源文生图服务，公开名称不能自动证明其内部是早期自回归路线、DALL·E 2 的 prior-decoder 路线、latent diffusion、DiT、flow matching，还是混合架构。书稿只能引用公开论文、官方说明或可复现实现，不能从产品输出反推全部内部组件。

### 10.5 两类路线的工程比较

| 维度 | 离散 token 自回归 | latent diffusion |
| --- | --- | --- |
| 状态 | 离散 image tokens | 连续像素或 latent |
| 训练目标 | next-token likelihood | epsilon、x0、v 或其他去噪目标 |
| 生成方式 | 按 token 顺序生成 | 迭代更新整个状态 |
| 主要成本 | token 序列长度和生成步数 | denoiser forward 次数和每步计算 |
| 条件方式 | prefix、cross-attention 或统一 token | text conditioning、CFG、control |
| 主要瓶颈 | tokenizer 和长序列 | sampler、VAE、latent 细节和多步延迟 |
| 优势 | 与语言模型统一 | 连续图像、编辑和控制生态成熟 |
| 边界 | 高分辨率序列长 | 少步采样和细节保真需要专门优化 |

不存在不带任务条件的绝对优劣。若目标是统一多模态 token 建模，自回归路线有吸引力；若目标是高质量图像生成和丰富的连续控制，latent diffusion 的工程生态更直接。

## 11. 文生图评估：不要用一个分数代表一切

### 11.1 评估维度

一套可用的评估至少分为：

1. 视觉质量；
2. prompt 遵循；
3. 属性和组合关系；
4. 空间布局；
5. 数量和文字；
6. 多样性；
7. 编辑保持率；
8. 安全与隐私；
9. 延迟、显存和成本。

不同维度可能互相冲突。提高 guidance 可能改善 prompt 遵循，却降低多样性；提高分辨率可能改善文字，却增加延迟；LoRA 可能提升角色相似度，却损害背景泛化。

### 11.2 FID 的适用边界

FID 在特征空间中比较真实图像和生成图像的均值与协方差。常见形式为：

~~~math
\mathrm{FID}
=
\|\mu_r-\mu_g\|_2^2
+
\operatorname{Tr}
\left(
\Sigma_r+\Sigma_g
-
2(\Sigma_r\Sigma_g)^{1/2}
\right).
~~~

它可以衡量某种总体分布距离，但不直接回答“图片是否遵循这个 prompt”。一个模型可能有不错的 FID，却漏掉 prompt 中的第二个对象。

### 11.3 CLIPScore 与语义对齐

CLIPScore 类指标使用图像和文本 embedding 的相似度作为参考无关的语义对齐代理。它适合提供一个切片信号，但有局限：

- 对数量和空间关系不敏感；
- 可能偏好常见语义；
- 不能保证文字正确；
- 不等于人类整体审美；
- 会受到 CLIP 模型和 prompt 表达影响。

因此 CLIPScore 适合和组合关系、人工偏好、文字识别及安全评估并列使用。

### 11.4 组合关系与 prompt benchmark

复杂 prompt 应拆成可核验的 claims，例如：

- 有一个红色自行车；
- 自行车在黄色房子旁边；
- 画面中有两只鸟；
- 文字出现在招牌上。

GenEval、T2I-CompBench 等公开研究可以作为组合、属性和关系评估的入口，但具体 benchmark 的覆盖、图像分布和判定器有边界。目标业务应建立自己的 prompt suite，并记录每个 claim 的成功事件。

### 11.5 文字专项评估

文生图中的文字需要独立测量：

- 字符可读率；
- OCR exact match；
- 字符编辑距离；
- 字号和字体切片；
- 文字位置；
- 多语言切片；
- 文字与背景对比度。

“海报整体好看”不能证明标题正确。对商品、票据和 UI 生成，文字错误可能比风格错误更严重。

### 11.6 编辑任务的保持率

image-to-image、inpainting 和 outpainting 要同时看：

- 新目标是否生成；
- 不应改变的区域保持率；
- 边界是否连续；
- 颜色、光照、透视是否一致；
- mask 外区域是否漂移。

可以用像素或特征差异做辅助，但编辑成功通常需要任务专门的结构和人工评估。

### 11.7 人工评估与单位成本

人工评估要固定 rubric、样本、随机化和盲测条件，至少区分质量、遵循、关系、文字、安全和偏好。不要让评审者只看一张图就给整体分。

生产还要计算单位成功任务成本。若第 i 个请求成本为 C_i，支持业务目标的成功事件为 S_i，则：

~~~math
C_{\mathrm{per\ success}}
=
\frac{\sum_i C_i}
{\sum_i\mathbf{1}[S_i=1]}.
~~~

要求成功事件分母大于零；评估窗口内没有一个满足业务目标的结果时，单位成功任务成本未定义，不能用零或无穷大掩盖“尚无成功证据”。如果任务要求“生成含正确文字的海报”，只有通过文字和版式检查的样本才能进入成功分母。只以“返回一张图片”计成功，会使成本数字失真。

## 12. 安全、版权与 provenance

### 12.1 内容安全是独立系统层

diffusion 数学只描述如何采样，不自动解决：

- 色情、暴力和违法内容；
- 名人肖像和 deepfake；
- 商标、品牌和身份冒用；
- 医疗、政治等高风险误导；
- 训练数据许可；
- 用户图片隐私；
- 生成内容的 provenance。

安全控制可能包含输入分类、输出分类、拒绝策略、人工复核、审计日志和水印，但每一层都应独立测量误报、漏报和绕过方式。

### 12.2 风格模仿与身份问题

“某艺术家风格”既涉及模型能力，也涉及数据、许可和产品政策。技术上可以把它当成条件或个性化任务，治理上还要判断是否涉及活跃艺术家、商业品牌、个人身份或误导性仿冒。

LoRA、文本 embedding 和 prompt 组合可以放大这种风险。训练和部署记录应包含数据来源、主体同意、删除路径和输出标记，而不是只记录权重文件名。

### 12.3 Provenance 不等于质量证明

水印、metadata 或生成标记可以帮助追踪来源，但它们不能证明图片内容真实，也不能证明没有被后处理。反过来，缺少 metadata 也不一定证明图片不是生成的。provenance、内容质量和安全判断是三个不同问题。

## 13. 贯穿案例：一张可控的机器人海报

### 13.1 业务请求

用户希望生成：

~~~text
一张横向水彩海报：红色机器人在花园里读书，黄色小屋在右侧，
标题写“周末阅读”，画面柔和、留出顶部文字空间。
~~~

这个请求同时包含：

- 对象：机器人、书、花园、小屋；
- 属性：红色、黄色、水彩；
- 空间关系：小屋在右侧；
- 文字：固定标题；
- 构图：顶部留白；
- 质量：柔和；
- 可能的安全和版权策略。

### 13.2 选择 pipeline

一种合理的系统设计是：

1. text encoder 读取语义和风格；
2. latent denoiser 生成整体构图；
3. ControlNet 或 layout condition 约束小屋和顶部区域；
4. inpainting 专门修复标题；
5. OCR 和版式检测验证“周末阅读”；
6. 失败时重新生成文字区域，而不是只提高 guidance。

如果模型对中文文字不稳定，直接要求 denoiser 一次性生成标题可能成本低却成功率差。分层 pipeline 可能多一次编辑和 OCR，但单位成功任务成本反而更低。

### 13.3 形状和成本

假设输出 768×512，VAE scale=8，latent channel=4：

~~~math
z\in
\mathbb{R}^{1\times4\times96\times64}.
~~~

latent 空间位置为 6144。若每步 CFG 需要正负两次 denoiser，30 步约有：

~~~math
N_{\mathrm{denoiser}}
=
30\times2=60.
~~~

如果每层 cross-attention 的 latent token 数为 6144、文本条件长度为 77，则单层单步 score cell 约为：

~~~math
C_{\mathrm{step}}
=
6144\times77
=
473088.
~~~

这不是 FLOPs 的完整值，却能说明分辨率、文本长度和采样步数如何共同进入预算。

### 13.4 失败诊断

如果机器人和小屋都出现，但标题乱码，优先检查文字专项生成和 inpainting；如果标题正确但小屋不在右侧，检查 ControlNet 或 layout 条件；如果每个 seed 都有相同构图，检查初始噪声和 guidance；如果图像颜色整体异常，检查 VAE scaling 和 decoder 范围；如果输出违反策略，检查输入/输出安全层，而不是继续调 scheduler。

### 13.5 评估记录

每次实验至少保存：

- prompt、negative prompt；
- seed；
- checkpoint、VAE、text encoder；
- scheduler、步数、guidance；
- 分辨率和 crop；
- ControlNet 类型、条件图和 scale；
- OCR 结果；
- 结构关系结果；
- 生成耗时和显存；
- 是否进入成功分母。

这样才能比较“更好的图片”究竟来自模型、采样器、seed 还是后处理。

## 14. 最小可运行 pipeline 审计

下面的程序只使用 Python 标准库，不生成图片。它把一个简化文生图请求转换成可手算的报告，检查：

- latent shape；
- latent 空间与像素空间比例；
- positive/negative text condition shape；
- CFG 造成的 denoiser 调用量；
- cross-attention 的累计 cell 数；
- image-to-image 的示例噪声时间步；
- ControlNet 条件 shape；
- 离散 image token 路线的序列成本。

其中 tokenizer 只是按空格和逗号切分的教学替代，不代表真实 tokenizer。ControlNet shape 也只是示意输入，真实 pipeline 可能在预处理后使用不同空间尺寸。

~~~python
import math
from math import ceil


def simple_tokenize(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [tok for tok in text.replace(",", " ").split() if tok]


def require_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def require_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def latent_shape(batch, height, width, vae_scale, latent_channels):
    for value, name in (
        (batch, "batch"),
        (height, "height"),
        (width, "width"),
        (vae_scale, "vae_scale"),
        (latent_channels, "latent_channels"),
    ):
        require_positive_int(value, name)
    return (
        batch,
        latent_channels,
        ceil(height / vae_scale),
        ceil(width / vae_scale),
    )


def elem_count(shape):
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("shape dimensions must be positive")
    total = 1
    for dim in shape:
        total *= dim
    return total


def audit_text_to_image_pipeline(config):
    for field in (
        "batch",
        "height",
        "width",
        "vae_scale",
        "latent_channels",
        "steps",
        "control_channels",
        "max_text_tokens",
        "text_hidden",
        "dalle_grid",
        "dalle_text_tokens",
    ):
        require_positive_int(config[field], field)
    require_nonnegative_int(config["special_tokens"], "special_tokens")
    if config["max_text_tokens"] < config["special_tokens"]:
        raise ValueError("max_text_tokens must include special tokens")
    if not math.isfinite(config["denoising_strength"]):
        raise ValueError("denoising_strength must be finite")
    if not 0.0 <= config["denoising_strength"] <= 1.0:
        raise ValueError("denoising_strength must be in [0, 1]")
    if not math.isfinite(config["guidance_scale"]):
        raise ValueError("guidance_scale must be finite")
    if config["guidance_scale"] < 0.0:
        raise ValueError("guidance_scale must be non-negative")
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
    if text_tokens <= 0 or negative_text_tokens <= 0:
        raise ValueError("text conditions must contain at least one token")

    z_shape = latent_shape(
        config["batch"],
        config["height"],
        config["width"],
        config["vae_scale"],
        config["latent_channels"],
    )
    latent_spatial_cells = z_shape[2] * z_shape[3]
    pixel_elements = (
        config["batch"] * config["height"] * config["width"] * 3
    )
    latent_elements = elem_count(z_shape)

    cfg_enabled = config["guidance_scale"] > 1.0
    denoiser_calls = config["steps"] * (2 if cfg_enabled else 1)
    condition_tokens_per_step = (
        text_tokens + negative_text_tokens
        if cfg_enabled
        else text_tokens
    )
    cross_attention_cells = (
        config["steps"] * latent_spatial_cells * condition_tokens_per_step
    )

    img2img_noise_step = int(
        round(config["steps"] * config["denoising_strength"])
    )
    control_shape = (
        config["batch"],
        config["control_channels"],
        config["height"],
        config["width"],
    )

    dalle_image_tokens = config["dalle_grid"] ** 2
    dalle_total_tokens = (
        config["dalle_text_tokens"] + dalle_image_tokens
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
        "spatial_ratio": round(
            latent_spatial_cells
            / (config["height"] * config["width"]),
            6,
        ),
        "element_ratio": round(
            latent_elements / pixel_elements,
            6,
        ),
        "denoiser_calls": denoiser_calls,
        "cross_attention_cells": cross_attention_cells,
        "img2img_noise_step": img2img_noise_step,
        "control_shape": control_shape,
        "dalle_total_tokens": dalle_total_tokens,
        "dalle_image_token_ratio": round(
            dalle_image_tokens / config["dalle_text_tokens"],
            3,
        ),
    }

    checks = {
        "latent_height_matches": (
            z_shape[2] * config["vae_scale"] >= config["height"]
        ),
        "latent_width_matches": (
            z_shape[3] * config["vae_scale"] >= config["width"]
        ),
        "cfg_doubles_denoiser_calls": (
            denoiser_calls == config["steps"] * 2
        ),
        "cross_attention_counts_conditions": (
            cross_attention_cells
            == config["steps"]
            * latent_spatial_cells
            * (text_tokens + negative_text_tokens)
        ),
        "latent_ratio_small": report["element_ratio"] < 0.05,
        "control_batch_matches": (
            control_shape[0] == config["batch"]
        ),
        "dalle_image_tokens_1024": dalle_image_tokens == 1024,
        "img2img_step_in_range": (
            0 <= img2img_noise_step <= config["steps"]
        ),
    }
    report["checks"] = checks
    return report


def expects_value_error(function):
    try:
        function()
    except (TypeError, ValueError):
        return True
    return False


config = {
    "prompt": (
        "a cinematic photo of a robot reading a book "
        "in a garden, natural light"
    ),
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

boundary_checks = {
    "zero_steps_rejected": expects_value_error(
        lambda: audit_text_to_image_pipeline({**config, "steps": 0})
    ),
    "invalid_strength_rejected": expects_value_error(
        lambda: audit_text_to_image_pipeline(
            {**config, "denoising_strength": 1.1}
        )
    ),
    "negative_guidance_rejected": expects_value_error(
        lambda: audit_text_to_image_pipeline(
            {**config, "guidance_scale": -1.0}
        )
    ),
    "non_finite_guidance_rejected": expects_value_error(
        lambda: audit_text_to_image_pipeline(
            {**config, "guidance_scale": float("nan")}
        )
    ),
    "non_finite_strength_rejected": expects_value_error(
        lambda: audit_text_to_image_pipeline(
            {**config, "denoising_strength": float("inf")}
        )
    ),
    "zero_latent_channels_rejected": expects_value_error(
        lambda: latent_shape(1, 512, 512, 8, 0)
    ),
    "non_string_prompt_rejected": expects_value_error(
        lambda: simple_tokenize(None)
    ),
    "empty_shape_rejected": expects_value_error(
        lambda: elem_count(())
    ),
}
assert all(report["checks"].values())
assert all(boundary_checks.values())
print(f"boundary_checks={boundary_checks}")
print(
    "all_checks_passed="
    f"{all(report['checks'].values()) and all(boundary_checks.values())}"
)
~~~

预期输出为：

~~~text
prompt_token_count=14
negative_token_count=6
text_condition_shape=(1, 16, 768)
negative_condition_shape=(1, 8, 768)
latent_shape=(1, 4, 64, 64)
latent_spatial_cells=4096
spatial_ratio=0.015625
element_ratio=0.020833
denoiser_calls=60
cross_attention_cells=2949120
img2img_noise_step=16
control_shape=(1, 1, 512, 512)
dalle_total_tokens=1040
dalle_image_token_ratio=64.0
checks={'latent_height_matches': True, 'latent_width_matches': True, 'cfg_doubles_denoiser_calls': True, 'cross_attention_counts_conditions': True, 'latent_ratio_small': True, 'control_batch_matches': True, 'dalle_image_tokens_1024': True, 'img2img_step_in_range': True}
boundary_checks={'zero_steps_rejected': True, 'invalid_strength_rejected': True, 'negative_guidance_rejected': True, 'non_finite_guidance_rejected': True, 'non_finite_strength_rejected': True, 'zero_latent_channels_rejected': True, 'non_string_prompt_rejected': True, 'empty_shape_rejected': True}
all_checks_passed=True
~~~

这个结果只证明示例配置在形状和算术上自洽：

- 512×512、scale=8 时得到 4×64×64 的 latent；
- 30 步、正负条件各一次时有 60 次 denoiser 调用；
- CFG 开启时，cross-attention cell 数同时计入正向和负向条件；
- 32×32 的离散图像 token 网格有 1024 个 image tokens；
- image-to-image 的噪声步仍需由目标 scheduler 的真实映射解释。

它不证明 prompt 一定被遵循，也不证明 ControlNet 真能保持姿态，更不证明生成内容安全。

## 15. 练习：从组件到系统判断

1. 画出 Stable Diffusion 类系统的训练和推理数据流，并标明 VAE、text encoder、denoiser 和 scheduler 的责任。
2. 对 1024×768、VAE scale=8、latent channel=4 的输入计算 latent shape 和元素比例。
3. 解释 latent scaling 为什么是 checkpoint 契约，而不是可有可无的常数。
4. 计算 latent token 数为 4096、文本 token 数为 77、cross-attention 层数为 8、采样步数为 30 时的累计 score cell 数。
5. 推导 CFG 公式，并说明 positive、negative 和 empty condition 的差别。
6. 设计一个实验区分 text encoder 截断和 denoiser 不遵循 prompt。
7. 说明为什么相同 seed 跨 VAE、scheduler 或精度版本不一定可复现。
8. 为 image-to-image 设计三个 denoising strength 切片，分别描述应观察什么。
9. 写出 inpainting 中 mask=1 表示重绘区域时的 latent 混合公式。
10. 说明 ControlNet 条件图的 crop、resize 和坐标错位会怎样影响结果。
11. 计算 LoRA rank、alpha 和目标矩阵维度下的可训练参数量。
12. 比较早期 DALL·E 自回归 image token、DALL·E 2 prior-decoder 和 latent diffusion 的条件路径。
13. 解释为什么 FID 不能单独代表 prompt 遵循。
14. 设计一组同时评估文字、数量、空间关系、多样性和安全的 prompt suite。
15. 对一个生成失败记录 checkpoint、VAE、text encoder、scheduler、seed、steps、guidance 和硬件，并设计最小消融。
16. 计算一批请求的单位成功任务成本，并说明哪些失败不能进入成功分母。

## 16. 资料入口与证据边界

1. Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models：<https://arxiv.org/abs/2112.10752>。支持 latent diffusion、VAE 压缩和条件 cross-attention 的公开研究路线。
2. Ramesh et al., Zero-Shot Text-to-Image Generation，早期 DALL·E：<https://arxiv.org/abs/2102.12092>。支持离散 image token 和自回归文生图路线。
3. Ramesh et al., Hierarchical Text-Conditional Image Generation with CLIP Latents，DALL·E 2：<https://arxiv.org/abs/2204.06125>。支持 text-to-CLIP-image-latent prior 与 diffusion decoder 的公开路线。
4. Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models，ControlNet：<https://arxiv.org/abs/2302.05543>。支持结构条件分支和残差控制的研究背景。
5. Peebles and Xie, Scalable Diffusion Models with Transformers：<https://arxiv.org/abs/2212.09748>。支持 DiT 作为 denoiser 主干的研究路线。
6. CompVis/stable-diffusion 官方开源仓库：<https://github.com/CompVis/stable-diffusion>。支持特定仓库版本的实现和配置核对，不代表所有社区 checkpoint。
7. Hugging Face Diffusers Stable Diffusion pipeline 文档：<https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview>。支持特定版本 pipeline API 的组件和参数入口。
8. Hessel et al., CLIPScore：<https://arxiv.org/abs/2104.08718>。支持图文相似度作为评估代理的研究背景，不等于完整人类偏好。
9. Huang et al., T2I-CompBench：<https://arxiv.org/abs/2303.13467>。支持组合属性和关系评估的公开 benchmark 入口。
10. Ghosh et al., GenEval：<https://arxiv.org/abs/2310.11513>。支持对象、属性和关系组合评估的研究背景。

原始论文支持方法与论文实验，官方文档支持版本化 API，仓库支持特定 commit，指标论文支持评估设计。本章的 pipeline demo 只支持 shape、token 和成本算术，不能证明真实图像质量、prompt 遵循、版权合规或安全效果。

## 17. 本章回顾

Stable Diffusion 类系统的核心不是一个神秘的“文生图网络”，而是一组模块契约：

~~~text
text encoder
    -> condition hidden states
VAE encoder or random latent
    -> noisy latent
denoiser + scheduler + CFG/control
    -> clean latent
VAE decoder
    -> image
~~~

VAE 决定像素和 latent 的转换，text encoder 决定 prompt 表示，denoiser 预测去噪参数，scheduler 决定采样轨迹，CFG 调整条件方向，ControlNet 等模块加入结构约束，inpainting 和 image-to-image 改变初始状态，LoRA 改变部分权重。

早期 DALL·E 的离散 image token 自回归路线、DALL·E 2 的 CLIP latent prior-decoder 路线和 latent diffusion 不是简单的版本号关系，而是不同的状态空间和条件组织方式。比较它们时，要比较训练目标、生成步骤、条件路径、成本和评估边界。

可靠的文生图判断必须同时查看 prompt 遵循、对象关系、文字可读性、编辑保持率、多样性、安全、延迟和单位成功任务成本。下一章将进入视频生成与 world model，讨论时间维度、帧间一致性、运动建模和视频评估。
