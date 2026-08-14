# 第十一章：Multimodal、Diffusion、Video 论文线

多模态模型处理的不再只是离散文本。图像、语音、视频和 3D 信号带有连续的空间或时间结构，语言则提供了便于组合、解释和控制的符号接口。研究者要解决的不是简单地“给语言模型加一张图”，而是如何表示不同模态、建立跨模态对应、融合信息、生成目标信号，并判断模型是否真正看见、听见和遵循了条件。

本章沿四条相互连接的论文线展开。第一条是图文对齐和视觉语言理解，以 CLIP、Flamingo、BLIP 和 LLaVA 为代表。第二条是扩散生成，以 DALL-E、Latent Diffusion、Stable Diffusion、EDM 和 ControlNet 为代表。第三条是语音基础模型，以 Whisper 的弱监督路线为代表。第四条是视频生成和时空建模，以 VideoPoet、视频扩散和 Sora 类公开技术路线为代表。

多模态研究最容易出现一种误判：模型能生成流畅描述，就被认为看懂了图；图像看起来漂亮，就被认为遵循了 prompt；视频第一帧合理，就被认为学会了运动；语音转写通顺，就被认为没有漏掉关键信息。可靠阅读必须把表示、对齐、融合和评估分开。

## 1. 多模态学习到底在对齐什么

### 1.1 从不同信号到共同任务

假设给模型一张图、一段声音和一条文字指令。图像包含像素、物体、位置和纹理，声音包含频率、时间、说话人和噪声，文字包含离散 token、语义和任务约束。它们的原始坐标系不同，不能直接比较。

多模态系统通常先把每种输入变成表示，再通过对齐损失或融合模块建立关系：

~~~text
image/audio/video
  -> modality encoder
  -> tokens / continuous features / latent
  -> alignment or projector
  -> cross-modal fusion
  -> understanding or generation head
~~~

这条链上的每一步都有自己的责任。编码器可能丢失小文字，projector 可能无法把视觉特征映射到语言空间，融合模块可能只依赖语言先验，生成器可能满足风格却忽略主体关系。把所有问题都叫“多模态能力不足”无法指导修复。

### 1.2 理解与生成是不同目标

多模态理解希望从信号中得到可判断的语义，例如分类、检索、OCR、VQA、语音识别和视频问答。多模态生成希望根据条件产生新的信号，例如文生图、图生图、文生视频、语音合成和视频补全。

| 任务 | 输入 | 输出 | 主要证据 |
| --- | --- | --- | --- |
| 图文检索 | 图像/文本 | 匹配度或排名 | Recall@K、排序质量 |
| VQA/OCR | 图像和问题 | 文本答案 | 正确性、定位和引用 |
| 语音识别 | 音频 | 转写文本 | WER、时间戳、说话人 |
| 文生图 | 文本 | 图像 | 文本一致性、质量、可控性 |
| 文生视频 | 文本 | 视频 | 时空一致性、运动和条件遵循 |

理解模型的输出可以是离散答案，生成模型的输出是高维信号。一个模型在图文检索上很好，不代表它能精确数图中物体；一个生成视频看起来真实，也不代表它正确执行了复杂动作约束。

### 1.3 四类架构路线

常见结构可以按融合位置分为：

1. **双塔对齐**：图像和文本分别编码，在共同 embedding 空间比较，CLIP 是代表。
2. **编码器到解码器**：视觉编码器提供特征，文本解码器生成 caption 或答案。
3. **视觉特征接入 LLM**：projector、adapter 或 cross-attention 把视觉 token 接到语言模型，Flamingo 和 LLaVA 属于这条大方向。
4. **统一生成模型**：把图像、视频或音频压缩成 latent 或离散 token，与文本条件一起用扩散或自回归 Transformer 建模。

架构名称不能代替数据和目标分析。同一种视觉 encoder 可以接在不同的语言模型上，结果差异可能主要来自 instruction data、冻结策略、分辨率和评估协议。

## 2. CLIP：用自然语言监督学习视觉表示

### 2.1 固定标签的限制

传统图像分类把目标限制在固定类别集合，例如 ImageNet 的标签。它能提供清晰监督，却很难覆盖开放世界中的细粒度概念、组合关系、风格和自然语言描述。每扩展一个新类别或新任务，都可能需要重新标注和训练。

互联网中存在大量图像和旁边的文字。文字质量不一定高，但它提供了比固定类别更开放的语义接口。CLIP 的核心问题是：能否使用大规模图文对，让图像 encoder 和文本 encoder 学到共同语义空间。

### 2.2 对比学习目标

给一个 batch 中的 $N$ 个图像和 $N$ 个文本，图像编码器得到 $v_i$，文本编码器得到 $t_j$。归一化后计算相似度：

~~~math

s_{ij}=\frac{v_i^{\top}t_j}{\lVert v_i\rVert\,\lVert t_j\rVert\,\tau},

~~~

其中 $\tau$ 是温度参数。正确配对在相似度矩阵的对角线上。图像到文本方向的交叉熵为：

~~~math

L_{I\rightarrow T}
=-\frac{1}{N}\sum_{i=1}^{N}
\log\frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})}.

~~~

文本到图像方向同理，通常使用对称目标：

~~~math

L_{CLIP}
=\frac{1}{2}
\left(L_{I\rightarrow T}+L_{T\rightarrow I}\right).

~~~

初学者可以把它理解为：正确图文在同一张“语义地图”上靠近，错误配对在 batch 内被推远。专家要注意，负样本通常来自 batch，batch 构成、重复图像、caption 噪声和温度都会改变对比信号。

### 2.3 Zero-shot classification

训练完成后，类别不必只用一个固定 ID 表示。对于类别 $c$，可以构造文本 prompt $p_c$，比较图像 embedding 与各个文本 embedding 的相似度：

~~~math

\hat c
=\arg\max_{c}
\operatorname{sim}(v(x),t(p_c)).

~~~

这就是 zero-shot 分类的直觉。模型没有在目标数据集上训练分类头，而是利用文字概念作为分类接口。

prompt 不是无关紧要的字符串。`a photo of a cat`、`an image of a cat` 和更具体的模板可能产生不同 embedding。公平评估要报告模板、类别名称、是否使用 prompt ensemble 和是否调过模板。

### 2.4 CLIP 的贡献与边界

CLIP 的主要贡献是把自然语言变成视觉监督接口，使视觉表示具备开放词汇迁移能力。它推动了图文检索、zero-shot 分类和后续文本条件生成路线。

但全局图文相似度不等于精确视觉推理。CLIP 可能知道图像大致是“猫和椅子”，却不可靠地判断猫在椅子左边还是右边，也可能在计数、OCR、细粒度属性和多对象关系上失败。

它还继承数据来源的问题：网页 caption 可能与图像无关，语言和视觉偏见会被一起学习，版权和隐私来源也需要治理。论文结果必须区分开放词汇语义对齐与细粒度 grounding。

### 2.5 CLIP 论文的复现实验

小规模复现可以使用公开图文数据，训练两个小 encoder，并比较：

1. batch size 对负样本数量和检索 Recall@K 的影响。
2. 温度参数对相似度分布和梯度的影响。
3. caption 噪声、重复和错误配对对检索的影响。
4. prompt 模板对 zero-shot 分类的影响。
5. 全局检索与局部定位任务的差异。

记录 image/text encoder、数据版本、归一化方式、温度、batch、训练步数和模板。否则“复现 CLIP”可能只是复现了一个相似的对比学习模型。

## 3. 视觉语言模型：从图文匹配到看图对话

### 3.1 视觉信息如何接入语言模型

视觉语言模型通常先用视觉 encoder 得到一组视觉特征 $V(x)$，再通过 projector 映射到语言模型 embedding 空间：

~~~math

Z_v=W_pV(x)+b_p.

~~~

这些视觉 token 可以与文本 token 拼接，也可以通过 cross-attention 注入语言层。语言模型随后根据图像特征和文字问题生成答案。

projector 的作用不是把图像“翻译成一句描述”，而是让下游语言模型能够消费视觉表示。若视觉 encoder 本身没有保留 OCR、空间和细粒度信息，projector 再大也无法凭空恢复。

### 3.2 Flamingo：冻结模型之间的桥接

Flamingo 研究如何把预训练视觉模型和语言模型连接起来，并处理交错的图像和文本输入。它通过适配结构让语言模型在多模态上下文中使用视觉信息，支持 few-shot multimodal learning。

阅读 Flamingo 类论文时，应分开看三件事：视觉 encoder 是否冻结，语言模型是否冻结，新增的 cross-attention 或 resampler 如何控制视觉 token 数。视觉 token 过多会增加语言模型上下文压力，过度压缩又会丢失细节。

Flamingo 的 few-shot 能力也依赖示例格式、模态顺序和任务分布。它证明的是给定多模态上下文接口下的迁移能力，不等于模型在任意图像任务上都具备精确 grounding。

### 3.3 BLIP：数据质量是结构的一部分

图文网页数据经常存在错配。图片旁的文字可能是广告、页面标题、评论或无关导航。BLIP 类工作把图文理解、caption 生成和数据过滤放在同一研究链中，说明多模态预训练的瓶颈不只在架构，也在监督质量。

一个简单的噪声检查可以比较：原始 caption 与图像相似度、模型生成 caption 的一致性、人工抽样和重复率。自动过滤会引入模型偏差，不能把过滤器分数直接当作真值。

### 3.4 LLaVA：视觉 instruction tuning

LLaVA 类路线通常由四部分组成：

1. 视觉 encoder，提取图像 patch 或全局特征。
2. Projector，把视觉特征映射到 LLM embedding 空间。
3. LLM，负责语言建模、对话和推理。
4. 视觉 instruction data，覆盖描述、问答、OCR、推理和多轮交互。

常见训练分两步。先训练或适配 projector，使视觉表示与语言空间连接；再进行视觉 instruction tuning，让模型学习“看到图以后怎样回答”。冻结哪些模块、使用多少视觉 token、图像分辨率和指令数据质量都会影响结果。

LLaVA 的重要性在于展示了一个可复用的工程接口：不必从零训练一个完全统一的视觉语言模型，也可以利用现成语言模型的推理和对话能力，再补上视觉输入和多模态监督。

### 3.5 视觉幻觉与语言先验

视觉语言模型可能回答图中不存在的物体，或者根据常识猜测而不是依据图像。可以把回答来源粗略分成：视觉证据、语言先验和指令模板。若图像模糊、低分辨率或视觉 encoder 没有保留局部信息，语言先验可能占上风。

一个简单的对照是：

1. 保持问题不变，替换图像为相似但关键事实不同的图。
2. 保持图像不变，改变问题措辞和先验诱导。
3. 要求模型指出证据区域或输出不确定性。
4. 比较有图和无图条件下的回答变化。

如果无图和有图答案几乎相同，流畅性不能证明模型真正使用了视觉输入。

### 3.6 Grounding、OCR 和空间关系

VLM 评估不应只看开放式回答。需要分别测量：

| 能力 | 典型问题 | 合适证据 |
| --- | --- | --- |
| Grounding | 物体在哪里 | box/point/region 一致性 |
| OCR | 图中写了什么 | 字符准确率、表格结构 |
| Counting | 有几个对象 | count accuracy |
| Spatial | 左右、前后、包含关系 | 关系准确率 |
| Chart/Doc | 读表、读图、引用字段 | 单元格和证据定位 |
| Multi-image | 两张图的差异 | 对应区域和属性变化 |

语言回答写得越长，不代表这些底层能力越强。grounding 证据应尽量来自坐标、区域或可验证的视觉事实。

## 4. Diffusion：学习从噪声恢复数据

### 4.1 生成模型的选择

VAE 通过潜变量重构数据，GAN 通过生成器和判别器对抗训练，自回归模型逐 token 生成，扩散模型则学习一系列逐步去噪操作。扩散路线的优势是训练目标相对稳定、条件控制灵活，代价是采样需要多次网络评估。

不要把“扩散模型”理解成一个单独网络结构。它包含前向加噪过程、反向去噪参数化、噪声 schedule、网络架构、采样器、条件注入和损失权重等一组选择。

### 4.2 前向加噪过程

给干净图像 $x_0$，在时间步 $t$ 加入高斯噪声。常见表达为：

~~~math

x_t
=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).

~~~

当 $t$ 增大，信号比例下降，$x_t$ 逐渐接近噪声。训练时模型学习从 $x_t$ 和时间条件 $t$ 估计噪声、速度或干净样本，具体 prediction target 取决于实现。

### 4.3 去噪目标

一种常见的噪声预测目标是：

~~~math

L_{diff}
=\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_{\theta}(x_t,t,c)
\right\|_2^2
\right],

~~~

其中 $c$ 可以是文本、类别、图像或其他条件。训练后，生成从随机噪声 $x_T$ 开始，按照采样 schedule 逐步得到 $x_0$。

这个公式解释了为什么扩散训练看起来像“学去噪”，但生成时却能产生新的图像。模型不是记住一张被加噪的图，而是在数据分布上学习不同噪声水平下的恢复方向。

### 4.4 DALL-E 与文本条件生成

DALL-E 路线展示了自然语言可以成为图像生成的条件接口。早期路线把图像压缩成离散 token，再与文本 token 一起自回归建模；后续系统结合图文表示和生成模型，改善文本到图像的质量与控制。

这条论文线的重要 claim 不是“文字能描述图像”，而是模型能否正确组合多个对象、属性、空间关系和风格约束。复杂 prompt 往往比单个对象更能暴露生成器的组合能力。

### 4.5 Latent Diffusion：在紧凑空间中去噪

像素空间的高分辨率图像维度很大。Latent Diffusion 先用 autoencoder 把图像压缩到 latent $z$，再在 latent 空间进行扩散：

~~~math

z=E(x),
\qquad
\hat x=D(\hat z).

~~~

U-Net 或 Transformer 在 latent 空间预测去噪方向，最后由 decoder 恢复像素图像。这样可以降低每个 diffusion step 的空间计算，但质量上限会受到 autoencoder 重构能力影响。

Stable Diffusion 的广泛传播与这条路线密切相关：它把文本编码器、latent diffusion、U-Net、cross-attention 和 VAE 解码组合成一个可使用的文生图系统。复现时要分别记录 VAE、text encoder、denoiser、sampler 和 guidance，而不是把所有结果归因于“扩散”。

### 4.6 Classifier-free guidance

无分类器 guidance 同时使用条件和无条件预测。设模型的噪声预测分别为 $\epsilon_{cond}$ 和 $\epsilon_{uncond}$，常见组合为：

~~~math

\epsilon_{guided}
=\epsilon_{uncond}
+s\left(\epsilon_{cond}-\epsilon_{uncond}\right),

~~~

其中 $s$ 是 guidance scale。$s$ 增大通常会强化 prompt 遵循，但也可能带来过饱和、伪影、模式变窄和细节失真。报告必须把 guidance 与采样步数、sampler 和随机种子一起记录。

### 4.7 ControlNet 与条件控制

自由文生图只提供文本条件，很多应用还需要边缘、深度、姿态、分割、草图、参考图或局部 mask。ControlNet 类方法为这些空间条件增加可训练控制分支，让生成结果保留原始结构的同时接受文本和视觉控制。

这条路线说明“生成质量”和“可控性”是不同指标。一个图像可以很漂亮却没有遵守姿态图，或者结构很准确但文本内容错误。评估要同时看条件遵循、视觉质量、可编辑性和多样性。

### 4.8 EDM：拆开扩散的设计空间

EDM 类工作把噪声参数化、网络预条件、采样 schedule、损失权重和 solver 设计系统地拆开。它的研究价值在于提醒读者：扩散实验中的收益可能来自采样器、预条件或噪声范围，而不一定来自主干网络。

复现扩散论文时应冻结并记录：

1. 数据预处理和分辨率。
2. noise schedule 与训练噪声分布。
3. prediction target 和 loss weighting。
4. network preconditioning。
5. sampler、步数和随机种子。
6. guidance、文本编码器和条件 dropout。

只比较最终图片而不保存这些配置，无法判断方法差异。

## 5. 图像生成评估：漂亮不是完整指标

### 5.1 分布质量与条件一致

FID 试图比较真实图像和生成图像在特征空间中的分布差异，常见形式为：

~~~math

FID
=\lVert\mu_r-\mu_g\rVert_2^2
+\operatorname{Tr}\left(
\Sigma_r+\Sigma_g
-2(\Sigma_r\Sigma_g)^{1/2}
\right).

~~~

它可以帮助衡量分布质量，却不直接证明图像遵循了每条文本约束。CLIP score 反映图文相似，却可能受到 encoder 偏差和 prompt shortcut 影响。人类偏好更接近真实体验，但主观、昂贵且需要明确标注标准。

### 5.2 组合和细粒度评估

文生图模型常在复杂关系上失败。评估应包含：

1. 多对象和属性绑定。
2. 数量和空间关系。
3. 文字、数字、标志和表格。
4. 参考图结构保持。
5. 局部编辑和 mask 边界。
6. prompt 变体与负面条件。

对于每个 prompt，保存随机 seed、采样器、步数和图像版本。人类评估要随机化模型顺序、盲化名称，并区分质量、条件遵循和安全性。

### 5.3 生成风险与数据证据

图像生成还要评估 PII、肖像、版权、风格模仿、暴力和误导性内容。模型能生成相似人物或风格，不等于训练数据中存在明确复制；反过来，缺少可见水印也不等于不存在数据来源风险。风险判断需要数据血缘、相似性测试、政策和人工审查共同支持。

## 6. Whisper：从弱监督音频到语音基础模型

### 6.1 语音与图像的不同

音频除了内容还有时间、说话人、口音、噪声、语速和重叠语音。把音频切成固定片段可能截断词语或改变上下文，长音频还需要分段、时间戳和拼接。

语音识别的目标通常是从音频 $a$ 生成文本 $y$。一个 encoder-decoder 形式可以用条件似然表示：

~~~math

L_{ASR}(\theta)
=-\mathbb{E}_{(a,y)}
\sum_{t=1}^{T_y}
\log\pi_{\theta}(y_t\mid a,y_{<t}).

~~~

与图像对比学习不同，语音识别直接要求模型输出有时间顺序的 token 序列，转写错误还会受到分词、专有名词和标点策略影响。

### 6.2 Whisper 的弱监督路线

Whisper 使用大规模互联网音频和转录文本进行多语言、多任务训练，展示了弱监督规模和任务提示可以带来较强迁移与鲁棒性。它的意义不是“噪声数据一定优于精标数据”，而是说明在合适的数据过滤、任务混合和模型规模下，弱监督可以覆盖大量口音、语言和场景。

读这类论文要检查音频来源、转写质量、语言分布、静音和音乐比例、切分方式以及评估语言是否与训练分布重叠。互联网转录可能包含错误、隐私和版权问题，规模不能替代治理。

### 6.3 WER 与分段影响

语音识别常用 word error rate：

~~~math

WER
=\frac{S+D+I}{N},

~~~

其中 $S$ 是 substitution，$D$ 是 deletion，$I$ 是 insertion，$N$ 是参考文本词数。中文、日文和无空格语言需要明确分词或字符级口径，否则不同报告无法直接比较。

WER 还会被分段策略影响。长音频切得过短可能丢失上下文，切得过长可能增加延迟和显存；时间戳错位可能让转写内容看似正确却无法用于字幕、检索或多模态对齐。

### 6.4 语音模型的失败边界

需要单独测试：

1. 低资源语言、方言和口音。
2. 噪声、音乐、回声和多人重叠。
3. 专有名词、数字、代码和医学术语。
4. 时间戳、说话人和长音频分段。
5. 语音翻译、语言识别和纯转写的任务切换。

一个平均 WER 很好的模型，可能在关键专有名词上承担很高风险。音频质量、说话人隐私和录音授权也属于实验边界。

## 7. Video：从空间生成到时空一致

### 7.1 视频不是独立图像的集合

视频生成要同时满足每帧质量和跨帧约束。设视频为 $x_{1:T}$，模型不仅要建模每帧空间结构，还要建模时间联合分布：

~~~math

p(x_{1:T}\mid c)
\neq
\prod_{t=1}^{T}p(x_t\mid c).

~~~

右侧的独立帧生成可能每帧都漂亮，却产生对象身份变化、闪烁、动作跳变和背景不一致。视频模型必须引入 temporal attention、3D convolution、时空 latent、帧间条件或自回归状态等机制。

### 7.2 视频生成的质量维度

视频评估至少应分为：

| 维度 | 要回答的问题 |
| --- | --- |
| 空间质量 | 单帧清晰、结构和细节是否合理 |
| 时间一致 | 物体身份、纹理和背景是否稳定 |
| 运动质量 | 动作、速度、轨迹是否自然 |
| 物理合理 | 遮挡、碰撞、重力和接触是否合理 |
| 条件遵循 | 主体、动作、场景和镜头是否符合文本 |
| 镜头连续 | 视角、景别、构图和叙事是否连贯 |

单一视频 FID 或单帧 CLIP score 无法覆盖这些维度。人类评估要明确比较视频长度、分辨率、帧率和播放条件。

### 7.3 视频模型的几条路线

主要路线包括：

1. 在图像 diffusion U-Net 中增加 temporal block。
2. 先生成关键帧，再做插帧、补全或运动扩散。
3. 使用图像模型初始化，再通过视频数据训练 temporal adapter。
4. 把视频压缩成离散或连续 token，用自回归 Transformer 建模。
5. 在时空 latent patch 上使用 Diffusion Transformer。

扩散路线通常需要多次去噪，质量和控制灵活；自回归路线可以统一多模态 token，却面临长序列和 codec 保真度；混合路线需要在空间质量、时间稳定和成本之间取舍。

### 7.4 VideoPoet 与统一多模态 token

VideoPoet 展示了把文本、图像、视频和音频等信号转成可组合 token，并使用 decoder-only Transformer 统一建模的路线。它的研究价值在于说明视频生成不必只沿着 U-Net 扩散路径，也可以继承语言模型的序列建模和条件组合方式。

这种模型的关键不是“用了 Transformer”一句话，而是 codec 如何压缩视频、不同模态 token 如何排列、训练任务如何混合、生成结果如何解码，以及长序列成本如何控制。若 codec 丢失运动细节，语言模型再强也无法恢复。

### 7.5 Sora 类公开信息的证据边界

Sora 类工作代表大规模视频生成向可扩展时空建模、镜头控制和世界动态模拟方向发展。公开技术报告可以支持它展示了哪些输入输出形式、样例能力和系统观察，但若训练数据、模型规模、架构细节、过滤流程和完整评估没有公开，就不能把社区推测当成事实。

讨论未完全公开的模型时，应把信息分成三层：

1. 官方页面或技术报告明确说出的内容。
2. 根据公开样例可以合理观察到的行为。
3. 没有公开证据支持的结构、数据和训练细节。

只有前两层可以进入事实性描述，第三层应明确标为未验证假设。生成样例也不等于可复现实验。

### 7.6 视频数据和训练成本

视频 token 数不仅随帧数增长，还随分辨率、压缩率、帧率和时空 patch 变化。一个视频训练预算可以粗略写成：

~~~math

D_{video}
=N_{clip}\,T\,H\,W\,r_{token},

~~~

其中 $N_{clip}$ 是 clip 数，$T$ 是帧数，$H,W$ 是空间尺寸，$r_{token}$ 是压缩后每个时空区域的 token 率。这个式子只是数据规模记账；实际计算还取决于 latent、attention、采样步数和重计算。

训练和评估还要处理视频版权、人物隐私、地点信息、生成误导和数据去重。一个公开样例不能说明训练数据合法，也不能说明生成视频没有隐私风险。

## 8. 多模态评估：从“看起来对”到证据可验证

### 8.1 模态证据矩阵

评估多模态模型时，可以把结果拆成向量：

~~~math

\mathbf{m}
=\left(
m_{align},m_{ground},m_{quality},m_{temporal},m_{safety},m_{cost}
\right).

~~~

图文模型重点关注 alignment、grounding、OCR 和推理；扩散模型重点关注质量、条件遵循、可控性和多样性；语音模型重点关注 WER、时间戳和公平性；视频模型还要加入 temporal consistency、motion 和物理合理性。

一个 aggregate score 只能用于某个明确权重下的选择，不能掩盖一个关键维度归零。例如医疗图像系统若 OCR 或 grounding 失败，即使语言流畅度很高也不能上线。

### 8.2 图文检索与 grounding

图文检索可以用 Recall@K、MRR 或 nDCG 测量排名，grounding 则要检查答案是否能定位到区域或时间段。检索正确不代表模型能说明依据，caption 正确也不代表它能处理空间关系。

评估应加入难负样本：同类但属性不同的图片、相同物体不同位置、文字相似但实体不同的文档和多图对比。否则模型只需依靠主题相似度就能获得高分。

### 8.3 生成质量与条件遵循

文生图和文生视频需要把 prompt 拆成实体、属性、动作、空间、时间和风格约束，逐项判断。一个图像整体很漂亮但漏掉数量约束，不能被一个整体审美分数掩盖。

可以对每条 prompt 建立属性级标签，报告每个属性的准确率、组合成功率和失败类型。人类评估要区分“看起来更好”与“更符合条件”，最好进行盲评和随机化。

### 8.4 语音和视频的时间指标

语音要报告语言/口音/噪声切片、WER 分母和时间戳误差。视频要报告帧间对象一致、运动连贯、镜头切换和文本条件沿时间的覆盖。

如果模型只在第一秒出现文本指定对象，后续对象消失，平均视频-文本相似度可能仍然不错。时间分段和轨迹级评估更能暴露这种失败。

## 9. Worked case：视觉语言模型看似会看图

某 VLM 在图像问答上得到很高的自动分数。人工审查发现：对没有图像的同一问题，模型也给出几乎相同的答案；当图中物体颜色和数量被修改时，回答很少变化。

这说明语言先验可能主导输出。可执行的对照包括：

1. 有图与无图配对。
2. 关键属性反事实编辑。
3. 只改变空间位置，保持对象和文字不变。
4. 要求输出证据区域或坐标。
5. 加入陌生对象和难负样本。

如果模型在整体 caption 上好、在属性反事实和 grounding 上差，结论应写成“具备主题描述能力，但视觉证据使用和细粒度 grounding 不足”。

## 10. Worked case：文生图的 prompt 遵循与视觉质量冲突

一个扩散模型在低 guidance scale 下图像自然、FID 较好，但漏掉 prompt 中的数量和空间关系；提高 guidance 后文本相似度上升，却出现饱和、伪影和多样性下降。

这不是简单的“调到更高 guidance 就更好”。应该固定 prompt、seed 和 sampler，扫描 guidance 与采样步数，分别记录条件属性准确率、图像质量、人类偏好、多样性和成本。若模型在复杂组合属性上始终失败，问题可能来自训练数据和文本编码器，而不是 scale 不够。

## 11. Worked case：视频第一帧正确但运动错误

某文生视频模型生成的第一帧包含“人把球放到桌上”，但后续球穿过桌面、手的形状变化，最后镜头切换到不相关场景。单帧 CLIP score 很高，视频整体任务成功率却低。

修复和评估应拆开：检查 temporal attention、视频长度、帧率、训练 clip、动作标签和物理约束；按帧记录对象身份，按轨迹评估球和手的位置，并对遮挡、接触和镜头连续性做人工标注。单帧指标不能替代时空一致性。

## 12. 可运行的多模态审计 demo

下面的纯 Python demo 使用合成记录，分别计算图文检索、grounding、扩散 prompt 遵循、语音 WER 和视频时间一致性。它不加载真实模型，只演示如何把“多模态能力”拆成独立信号。

~~~python
def mean(values):
    return sum(values) / len(values) if values else 0.0


def wer(substitutions, deletions, insertions, reference_words):
    return (substitutions + deletions + insertions) / reference_words


records = {
    "retrieval": [1, 1, 0, 1],
    "grounding": [1, 0, 1, 0],
    "prompt_attributes": [1, 0, 1, 0],
    "video_temporal": [1, 1, 0, 0],
}
signals = {
    "image_text_recall_at_k": round(mean(records["retrieval"]), 4),
    "grounding_accuracy": round(mean(records["grounding"]), 4),
    "prompt_attribute_accuracy": round(mean(records["prompt_attributes"]), 4),
    "video_temporal_consistency": round(mean(records["video_temporal"]), 4),
    "speech_wer": round(wer(2, 1, 1, 20), 4),
}

actions = []
if signals["grounding_accuracy"] < 0.75:
    actions.append("add_region_and_counterfactual_vision_tests")
if signals["prompt_attribute_accuracy"] < 0.75:
    actions.append("add_compositional_prompt_eval")
if signals["video_temporal_consistency"] < 0.75:
    actions.append("add_track_and_motion_review")
if signals["speech_wer"] > 0.15:
    actions.append("review_audio_slices_and_segmentation")
decision = "continue_after_modality_repairs" if actions else "continue_to_holdout_eval"

for name, value in signals.items():
    print(f"{name}={value}")
print(f"actions={actions}")
print(f"decision={decision}")
~~~

这个 demo 故意让总体检索结果看起来尚可，但 grounding、属性组合、视频一致性和 WER 暴露不同问题。真实实验还要加入模型、分辨率、采样器、语言切片、人工评估、版权/隐私审计和成本。

## 13. 常见失败模式与诊断

### 13.1 VLM 输出流畅但不依赖图像

做有图/无图对照、关键属性反事实和区域 grounding。检查视觉 token 数、分辨率、projector 训练和语言先验。不要把回答长度或语法质量当作视觉使用证据。

### 13.2 OCR 和表格理解差

检查图像分辨率、裁剪、文字方向、表格结构和视觉 encoder 的 patch 尺度。将整图 caption 与局部区域 OCR 分开评估，必要时使用专用 OCR 或结构解析，不要只扩大语言模型。

### 13.3 扩散图像漂亮但 prompt 关系错误

做属性级和组合级评估，扫描 guidance、采样器、步骤和 prompt 模板。若关系错误稳定存在，应检查数据 caption 和模型表示，而不是只优化审美评分。

### 13.4 视频闪烁和身份漂移

按对象和时间轨迹评估，检查视频 clip 长度、temporal module、latent codec 和帧率。单帧增强可能反而破坏跨帧一致，必须保留时序回归集。

### 13.5 语音低平均 WER 但关键词错

按语言、口音、噪声、专有名词和领域切片，报告词级错误和时间戳。医疗、法律和会议转写不能只看总体平均。

### 13.6 生成模型数据和版权边界不清

保存数据来源、许可、去重、删除和相似性测试；将模型事实、样例观察和法律判断分开。工程相似度指标不能自动推出法律结论。

## 14. 复现多模态论文的最小路线

### 14.1 先写窄 claim

不要从零复现“多模态模型更强”。可以选择：

> 在相同图文数据和 batch 下，温度或负样本规模是否影响图文 Recall@K？

或：

> 在固定 VLM 和语言提示下，加入区域 grounding 监督是否提高属性反事实准确率而不损害开放式问答？

或：

> 在相同扩散 checkpoint 和采样预算下，guidance scale 如何改变 prompt 属性遵循与视觉质量？

窄 claim 才能把结构、数据、采样和评估责任分开。

### 14.2 固定实验对象

至少保存：

1. 数据 snapshot、图像/音频/视频许可和去重结果。
2. tokenizer、文本模板、图像分辨率、音频采样率和视频帧率。
3. encoder、projector、decoder、VAE/codec 和 checkpoint。
4. batch、学习率、温度、noise schedule、sampler、steps 和 guidance。
5. 随机 seed、生成数量、评估脚本和人工标注协议。
6. GPU、显存、延迟、生成成本和失败样本。

多模态数据的预处理经常决定结果。若图像 crop、音频切分或视频帧采样不同，不能把数值差异简单归因于模型结构。

### 14.3 复现层级

可以将复现分为：

1. 接口复现：输入、输出和代码能够运行。
2. 数值复现：loss、检索指标、WER 或图像指标接近。
3. 行为复现：grounding、条件遵循、时间一致性等现象重现。
4. 主张复现：论文中的核心比较在公平预算下成立。

没有原始数据、模型权重或评估脚本时，应明确称为机制复现或替代数据复现。视频和大规模图文论文尤其不能用几个样例替代统计证据。

## 15. 研究报告：把模态事实写清楚

一份多模态论文复现报告应交代：

1. 输入模态和表示方式。
2. 对齐目标和负样本/条件构造。
3. 融合位置、冻结策略和可训练参数。
4. 生成空间、噪声或 codec 参数。
5. 评估指标、分母、位置/语言/时间切片。
6. 自动指标与人工判断的关系。
7. 质量、条件遵循、grounding、时序、安全和成本结果。
8. 失败样本、数据噪声、版权/隐私和未公开信息。

例如，不能把 CLIP Recall@1、图像 FID、语音 WER 和视频人类偏好平均成一个“多模态分数”。这些指标回答不同问题，除非任务明确给出权重，否则应并列报告。

## 16. 资料与证据边界

图文对齐可参考 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)。该论文支持图像/文本双塔、对比学习和 zero-shot 迁移的技术事实；具体开放世界泛化、偏见和版权风险需要在目标数据上单独评估。

视觉语言模型可参考 [Flamingo](https://arxiv.org/abs/2204.14198)、[BLIP](https://arxiv.org/abs/2201.12086) 和 [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)。它们支持交错图文 few-shot、图文数据治理和视觉 instruction tuning 的研究路线；本章关于 grounding、OCR 和幻觉的结论需要专项测试。

扩散和图像生成可参考 [DALL-E](https://arxiv.org/abs/2102.12092)、[DALL-E 2](https://arxiv.org/abs/2204.06125)、[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)、[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) 和 [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)。这些论文支持离散图像 token、文本条件、latent diffusion、扩散设计空间和结构控制的技术脉络；FID、CLIP score 和人类偏好不能互相替代。

语音识别可参考 [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)。论文支持大规模弱监督、多语言和鲁棒语音识别路线；具体语言、口音、专有名词、隐私和部署 WER 需要独立测量。

视频和时空生成可参考 [VideoPoet](https://arxiv.org/abs/2312.14125) 与 [Video generation models as world simulators](https://arxiv.org/abs/2402.17177)。它们支持多模态 token 建模和大规模视频生成的公开研究方向；对于未公开的数据、模型细节、训练规模和能力原因，本章只保留公开事实和明确的未验证边界。

评估和系统实验中的数值、worked case、Python demo 和决策字符串均为教学构造。真实项目应保存模态数据血缘、授权、预处理、模型配置、采样/解码参数、人工标注、失败样本和成本，并区分论文事实、官方样例、项目实测与合理推测。

## 17. 结语：多模态不是把名词堆在一起

多模态论文的基本问题可以用四个词串起来：表示、对齐、融合、生成。CLIP 通过图文对比学习建立共同空间；Flamingo、BLIP 和 LLaVA 研究视觉特征如何进入语言模型并形成对话；扩散模型学习从噪声恢复图像或视频，latent 和 guidance 改变成本与控制；Whisper 说明弱监督可以扩展语音基础能力；VideoPoet 和 Sora 类方向把问题推进到长序列时空建模。

但任何“看起来会”的结果都要追问证据：模型是否真正使用了视觉或音频，图像是否遵循了每个条件，视频是否保持对象和运动，语音是否在关键切片上可靠，数据和生成结果是否可治理。

当表示、对齐、融合、生成、评估和证据边界都被分别说明时，多模态才是一个可以复现、诊断和部署的研究对象，而不只是一串漂亮的模型名称。
