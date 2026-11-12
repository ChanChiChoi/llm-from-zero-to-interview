# 第十三章：Encoder-Free 统一多模态——先把术语说清楚，再讨论架构

“Encoder-free unified multimodal”听起来像一个非常明确的技术结论：模型没有视觉编码器，也没有音频编码器，所有模态直接进入同一个 Transformer。但在真实论文、模型卡和产品宣传中，`encoder-free` 的含义并不总是一致。有的作者指的是没有独立的视觉语言 encoder；有的指的是图像被离散 tokenizer 直接变成 token；有的只是强调理解和生成共享主干；还有的系统仍然使用 patchifier、codec、embedding、projection 或模态适配器。

因此本章先解决词义，再讨论结构。我们会把“前端编码器”“模态 tokenizer”“统一主干”“输出 decoder”拆开，说明它们各自承担什么责任；随后讨论统一 token 流的序列化、位置和 mask、训练目标、信息压缩、评估、部署协议和安全边界。最后用一个可运行的审计示例，检查不同模态 token、位置、预算和反事实实验是否自洽。

第十章已经建立统一架构的总地图；本章只把镜头收窄到表示如何进入主干，以及 token、位置、可见性和处理器如何构成可审计的数据路径，不重复比较所有统一架构，也不替代第十四章的请求预算分析。阅读时可沿三段主线推进：§§13.1–13.6 拆解前端与序列表示；§§13.7–13.12 讨论压缩、训练和部署协议；§§13.13–13.15 用评估与故障定位检验前面的设计。§13.16 的代码是清单检查示例，不是模型实现或性能实验。

## 13.0 从一个表格截图开始

用户上传一张财务表格，问：“第三行的数值比第二行高多少，并解释原因。”一个传统模块化系统可能这样处理：

```text
图片 -> vision encoder -> projector -> LLM -> 文本答案
       OCR --------------------------->|
       表格结构解析 ------------------->|
```

一个更统一的系统可能把图像 patch、OCR span、行列坐标、问题和目标输出放在一条混合序列中：

```text
image patches + OCR spans + coordinates + question
                         -> shared multimodal backbone
                         -> numeric verifier -> answer
```

第二种方案并不自动更好。它可能减少显式模块之间的桥接，却仍然需要图像 patchifier、文字 tokenizer、坐标编码、数值验证器和输出解码器。真正要问的是：

1. 哪些信息进入共享主干？
2. 哪些信息在进入主干前已经被压缩？
3. 主干能否区分图像、OCR 和坐标的来源？
4. 生成文本、图像和工具动作是否使用相同的目标？
5. 发生错误时能否回放输入、token、位置和 mask？

## 13.1 “Encoder-Free”不是“没有前端处理”

### 13.1.1 传统模块化路径

在常见视觉语言模型中，图像先交给独立视觉编码器，得到视觉 hidden states，再通过 projector、Q-Former 或 resampler 接入语言模型：

~~~math
H_v=E_{\mathrm{vision}}(x_v),
\qquad
Z_v=H_vW+b
~~~

`x_v` 是图像，`E_vision` 是视觉编码器，`H_v` 是视觉表示，`Z_v` 是对齐到语言主干 hidden size 的表示。语音和视频也有类似的专用 encoder、codec 或时空模块。

这里的线性写法省略了 batch 维度和可能存在的非线性 connector。若 `H_v\in\mathbb{R}^{B\times N_v\times d_v}`，则通常要求 `W\in\mathbb{R}^{d_v\times d_l}`、`b\in\mathbb{R}^{d_l}`，输出为 `B\times N_v\times d_l`。shape 能对上只说明矩阵乘法合法，不说明视觉坐标、归一化、token 顺序和语义已经正确。

这类架构的优势是模态专用模块可以针对信号结构优化，缺点是桥接层可能压缩细节，理解和生成路径也可能分裂。

### 13.1.2 可能被称为 encoder-free 的几种情况

下面几种情况都可能在资料中被称为更“encoder-free”，但含义不同：

1. 没有单独的视觉语言 encoder，图像先由离散 tokenizer 变成 token。
2. patch embedding 和主干 Transformer 由同一套训练目标联合优化。
3. 理解和生成共享一个主干，输出端仍有模态 decoder。
4. 不使用传统的预训练视觉 encoder，但仍有 patchifier、codec 或 projection。
5. 输入输出都以统一序列形式表示，模块边界被放到 tokenizer 和 decoder。

因此正确的教材表述应是：“该路线减少或取消了某一类独立模态 encoder 的责任，并把更多建模工作交给统一 token/latent 主干。”不能直接写成“完全没有前端编码器”。

### 13.1.3 为什么要区分 encoder、tokenizer 和 patchifier

三者经常被混用：

- `patchifier`：按空间或时间切分输入，可能只做局部线性映射。
- `tokenizer`：把输入映射为离散 token 或连续 latent，通常包含压缩和可逆/近似可逆结构。
- `encoder`：从输入提取较高层的上下文表示，可以包含多层 attention、卷积或时空建模。

一个系统没有独立 vision encoder，并不表示它没有图像 tokenizer；一个系统使用统一 Transformer，也不表示原始像素直接作为普通文本 token 输入。阅读结构图时，应把这些模块分别标出来。

## 13.2 统一的三个层次

### 13.2.1 接口统一

客户端使用同一套消息格式传递文本、图片、音频、视频和工具事件。接口统一只说明输入输出协议，不说明内部参数是否共享。

### 13.2.2 表示统一

公式中的 g_m 是中性的模态前端记号，可以代表 patchifier、离散 tokenizer、codec、专用 encoder 或它们的组合；它不预设前端是否属于独立的深层 encoder。

不同模态被编码到可以互相访问的序列或 latent 空间：

~~~math
z_m=g_m(x_m)\in\mathbb{R}^{N_m\times d}
~~~

这里的 `d` 统一了 hidden size，但不等于每种模态的信息语义已经完全对齐。图像 token 仍然需要二维位置，视频 token 仍然需要时间索引，音频 token 仍然有采样率和 codebook 层级。

这个抽象至少要求 `N_m` 是非负整数、`d>0`，并且每种 g_m 前端的输出 dtype、设备和特殊 token 约定可以被主干接受。`N_m` 相同也不意味着 token 可以互换：一个图像 token 的坐标、一个音频 token 的时间和一个工具结果的权限来源，仍然需要元数据和 mask 共同解释。

### 13.2.3 目标统一

理解、编辑、生成和工具动作共享一部分训练或推理路径。目标统一不等于所有模态使用同一个 loss。文本可能使用 next-token prediction，图像可能使用离散 token 或 diffusion，工具动作还需要 schema 和权限检查。

## 13.3 统一序列的数学抽象

设文本、图像、音频和视频分别经过模态处理得到序列：

~~~math
Z_t=(z_{t,1},\ldots,z_{t,N_t}),\quad
Z_i=(z_{i,1},\ldots,z_{i,N_i}),
~~~

~~~math
Z_a=(z_{a,1},\ldots,z_{a,N_a}),\quad
Z_v=(z_{v,1},\ldots,z_{v,N_v})
~~~

通过 interleave、拼接或块状布局形成统一序列：

~~~math
Z=\mathrm{Interleave}(Z_t,Z_i,Z_a,Z_v,Z_s)
~~~

`Z_s` 是模态边界、角色、时间戳、坐标和版本等特殊或元数据元素。统一主干计算：

~~~math
H=\mathrm{Transformer}(Z;M,\Pi)
~~~

`M` 是 attention visibility 或 causal mask，`Pi` 是位置结构。这个抽象表达的是共享信息路由，不是所有模态都使用相同 tokenizer、相同位置编码或相同输出头。

若统一序列为空，主干是否允许空上下文需要单独定义；若 `Z_s` 没有记录媒体边界、角色或版本，交错顺序本身不足以恢复来源。对同一个请求，序列化器还应保证 token、位置和 mask 使用同一索引，否则“看见了正确 token”仍可能等价于读取了错误区域。

把数据路径画成图，可以看出“统一”并没有删除每个环节的责任：

~~~mermaid
flowchart LR
  X["文本、图像、音频、视频"] --> F["模态前端：tokenizer、patchifier、codec"]
  F --> Z["混合序列：token、边界、来源"]
  Z --> P["位置结构：一维、二维、时间与空间"]
  Z --> A["任务 mask：定义可见关系"]
  P --> B["共享主干"]
  A --> B
  B --> D["任务输出头或 decoder"]
~~~

读图时，先从输入沿前端进入序列；位置字段说明 token 对应的时间、区域或顺序，mask 再规定当前任务允许哪些位置互相读取。它们作为不同信息共同进入主干，并不意味着位置编码就是 mask，也不意味着输出端共用一个 decoder。部署回放需要把这些字段和各自的 revision 一起保存。

### 13.3.1 序列布局的三种方式

块状布局把同一模态 token 放在一起：

```text
[text][image][audio][video][answer]
```

交错布局按照对话或媒体出现顺序排列：

```text
[text][image][text][image][answer]
```

层级布局先在模态内部建模，再让统一主干访问压缩后的摘要：

```text
[image summary][video event summary][question][answer]
```

块状布局容易实现，交错布局适合混合生成，层级布局节省上下文却可能损失局部证据。布局应由任务和训练数据决定，而不是把某一种排列当成统一模型的定义。

## 13.4 位置结构：一维序列不等于一维世界

### 13.4.1 文本、图像和视频的位置不同

文本通常使用一维位置 `p`；图像 token 需要二维位置 `(h,w)`；视频 token 至少需要三维位置 `(t,h,w)`；音频需要时间位置，可能还需要 codebook 或频带信息。

~~~math
\pi_{\mathrm{text}}=p,
\qquad
\pi_{\mathrm{image}}=(h,w),
\qquad
\pi_{\mathrm{video}}=(t,h,w)
~~~

如果把所有 token 只标成从 1 到 `N` 的整数，模型可以知道序列顺序，却不一定知道图像左上角、视频第 13 秒或音频第 4 个 codebook 的含义。

### 13.4.2 坐标和时间戳属于证据

合同问答中，`payment_term` 的文本内容和其页面坐标应绑定；视频问答中，事件描述和时间段应绑定；语音转写中，词和时间戳应绑定。可把一个证据元素记为：

~~~math
e_j=(z_j,\pi_j,s_j,c_j)
~~~

其中 `z_j` 是 token 或 span，`pi_j` 是位置，`s_j` 是来源，`c_j` 是置信度。把 `s_j` 和 `pi_j` 丢掉，后面的模型可能仍能生成流畅文本，却无法回答“哪一页、哪一帧、哪一个区域”。

## 13.5 Mask：统一信息流的真正控制面

### 13.5.1 不是所有 token 都应该互相可见

图像理解时，问题 token 可以访问整张图的视觉 token；视频自回归生成时，当前时间位置不能看到未来帧；工具结果可以作为外部观察，但不能覆盖系统策略。可把可见性写成矩阵 `M`：

~~~math
M_{uv}=\begin{cases}
0,&\text{位置 }u\text{ 可以读取 }v\\
-\infty,&\text{位置 }u\text{ 不可以读取 }v
\end{cases}
~~~

在 attention logits 上加入 `M` 后，不可见位置的权重被压到零。不同任务可能需要不同 mask，不能只保存一个“统一模型 mask”。

这一定义要求每个 query 至少有一个可见 key；如果一整行都是 `-\infty`，softmax 的分母为零，常见实现会产生 `NaN`。`M` 的数值语义也只适用于加到 logits 的实现，布尔 mask、稀疏索引和 block mask 需要在接口层明确约定。因果 mask 还必须与训练时的 shift 和推理时的 KV cache 使用同一时间方向。

### 13.5.2 常见 mask 错误

1. 文本答案看不到图像 token。
2. 生成目标提前看到未来 token，离线分数虚高。
3. 多张图片没有媒体边界，问题把上一张图当成当前图。
4. OCR span 可以覆盖系统消息的优先级。
5. 工具结果和用户指令混在同一信任层。

mask 错误通常不会导致程序崩溃，却会让模型行为悄悄改变。因此 golden request 必须保存位置和 mask，而不仅仅保存原始媒体。

在一个三位置的教学 attention 例子中，原始 logits 为 [0, ln 3, 2]，第三个 key 是未来视频帧。给第三项加上 mask 值负无穷后，softmax 变为 [exp(0), exp(ln 3), 0] 的归一化结果，即 [1/4, 3/4, 0]。模型仍能在前两个可见位置间分配注意力，但对未来帧的权重严格为零。这个小算例说明 mask 施加在 logits 后、softmax 前；生产实现可能用布尔、稀疏或 block mask 表示同一可见关系，具体约定应核对框架接口。

## 13.6 Token 化路线：连续、离散和混合

### 13.6.1 连续表示

本节公式中的 H_m 是前端 g_m 对输入处理后的连续输出；g_m 可能是浅层投影或专用 encoder，本章不从表示形式单独推断它的层数。

连续表示路径中的模态前端输出浮点 hidden states；此前端可以是浅层投影，也可以包含专用 encoder：

~~~math
H_m\in\mathbb{R}^{N_m\times d_m}
~~~

连续表示保留可供后续网络处理的几何关系，常用于理解和 cross-attention；生成端则可能需要专用 decoder、扩散过程或声码器。

### 13.6.2 离散表示

公式中的 g_m 先形成待量化的表示，Q 再把它映射到离散码本。g_m 可能只是局部投影，也可能是视觉或音频 encoder；只有资料明确说明取消了哪一个独立 encoder，encoder-free 才有可核查的含义。

离散 tokenizer 把输入映射成有限词表中的整数：

~~~math
z_m=Q(g_m(x_m)),
\qquad
z_m\in\{1,\ldots,V_m\}^{N_m}
~~~

`Q` 表示量化，`V_m` 是模态词表大小。离散 token 可以直接进入 next-token 模型，却会引入量化误差和长序列成本。

公式使用一基索引 `\{1,\ldots,V_m\}` 只是数学记号；实际程序常用 `0,\ldots,V_m-1`，还可能保留 `BOS`、`EOS`、`PAD` 和媒体边界等特殊 id。无论采用哪一种约定，都必须固定词表大小、保留 id 和越界行为。`V_m` 必须为正整数，量化结果不能含有 NaN 或超出词表的整数。

### 13.6.3 混合表示

Transfusion 展示了共享模型中同时使用语言建模和图像 diffusion 的路线；Show-o 结合自回归与离散扩散；Emu3 则展示了把图像、文本和视频离散化后使用 next-token 训练的路线。它们说明“统一主干”和“统一生成目标”是两个不同维度。

## 13.7 信息压缩和任务失真

统一 token 流的关键代价是压缩。设输入媒体为 `x`，编码后为 `z`，解码或重建为 `x_hat`，媒体失真为 `d`：

~~~math
D(R)=\mathbb{E}[d(x,\hat{x})]
~~~

`R` 可以理解为 token rate 或表示容量。工程真正关心的是任务失真，而不是只有像素或波形失真：

- 小数点丢失会改变合同金额。
- 行列关系丢失会改变表格比较。
- 时间戳丢失会改变事件顺序。
- 说话人信息丢失会改变身份判断。

因此要绘制 token 数与任务质量曲线，而不能因为序列更短就认为架构更好。

`D(R)` 只有在样本分布非空、`d(x,\hat{x})` 可计算且重建输出与输入处于同一比较空间时才有定义。像素均方误差、波形误差、感知距离和任务错误率不是同一个 `d`，不能在没有归一化和样本说明时直接比较。更重要的是，低媒体失真并不保证关键字段或权限信息没有丢失，因此应同时记录任务级失真。

举一个明确标为教学假设的文档例子：同一组 10 个关键字段，保留 64 个视觉 token 时有 9 个字段读对，字段错误率是 (10-9)/10=0.10；压到 16 个 token 后只读对 7 个，错误率变成 (10-7)/10=0.30。token 数减少了 48/64=75%，任务错误率却从 10% 升到 30%。这不是某个模型的实测曲线，只说明压缩评价必须同时给出 rate 和任务失真，且比较时固定样本、字段分母和处理器。

压缩曲线要按任务拆分，而不是只报一个平均相似度。合同字段、图表数值、说话人、事件先后和动作参数各自有不同的错误代价；同一份媒体在不同任务上也可能需要不同 token 数。做对照时固定模型 revision、processor、媒体集合、提示和评分规则，并同时报告每档 token 预算下的任务分母、错误类型与置信区间。这样才能判断省下的表示容量是否值得承担新增的任务错误，而不是用一个像素指标替所有下游目标做决定。

## 13.8 训练目标和共享主干

统一模型的总损失可以抽象成：

~~~math
\mathcal{L}
=\lambda_{\mathrm{text}}\mathcal{L}_{\mathrm{text}}
+\lambda_{\mathrm{media}}\mathcal{L}_{\mathrm{media}}
+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}
+\lambda_{\mathrm{safe}}\mathcal{L}_{\mathrm{safe}}
~~~

不同模态的有效 token 数、样本长度和 loss 尺度不同，`lambda` 不能脱离归一化口径解释。一个音频样本有上千 codec token，一条文本样本有几十 token；若直接累加逐 token loss，音频可能获得隐式更大权重。

通常要求每个 `\lambda` 非负且有限，并至少有一个正权重；每个子损失也应在非空有效样本或 token 上定义。如果某个 batch 没有任何安全标签或没有 assistant token，不能把该项 loss 默认为零后悄悄改变训练比例。报告权重时还要写明是按样本、按有效 token，还是先按模态求均值再混合。

考虑一个只用于演算的 batch：文本有 2 个有效 token，平均交叉熵为 (0.2+0.6)/2=0.4；音频有 6 个有效 token，平均交叉熵为 0.8。若将 8 个 token 的 loss 直接合并，结果是 (2×0.4+6×0.8)/8=5.6/8=0.7；若先分别求模态均值、再给两个模态相同权重，结果是 (0.4+0.8)/2=0.6。两种口径都可以定义，但第一种让 token 较多的音频占更大份额。报告总损失时必须同时说明分母和模态权重，才能解释训练信号。

逐 token 平均可写成所有有效 token loss 的总和除以有效 token 数；按模态宏平均则先对每种模态分别归一，再求带权平均。上面的 0.7 与 0.6 来自分母不同，并非某一算术错误。真实训练还会受到样本采样比例、序列截断、loss mask 和缺失标注影响；调小音频 loss 权重无法补回被采样器少取的音频样本，增加音频样本也不一定改变逐 token 梯度的隐式权重。复现实验时需同时记每模态样本数、有效 token 数和参与 loss 的目标数，才能区分“数据没见到”和“见到了但梯度较弱”。

### 13.8.1 理解和生成的张力

理解任务要保留可验证事实，生成任务要追求连续性、自然度和条件遵循。共享主干可能产生能力迁移，也可能发生梯度冲突。应分别记录：

1. 哪些参数共享。
2. 哪些输出 head 专用。
3. 哪些 token 参与哪一个 loss。
4. 多模态数据的采样比例。
5. 文本能力和媒体能力的回归集。

### 13.8.2 多阶段和联合训练

多阶段训练先稳定模态表示，再训练连接和跨模态任务；联合训练能让模型直接学习跨模态关系，却更容易受到数据比例和目标冲突影响。选择取决于数据规模、算力、可解释性和上线风险，不能用“端到端”三个字代替训练方案。

## 13.9 统一主干的模态干扰

共享容量会产生竞争。高分辨率图像可能占满上下文，重复视频帧可能稀释问题，错误 OCR 可能成为语言捷径。可以使用模态消融诊断：

```text
image only
OCR only
image + correct OCR
image + incorrect OCR
low-resolution image
high-resolution image
```

若加入错误 OCR 后答案稳定跟随 OCR，说明模型依赖文字捷径；若高分辨率只提高小字识别，却显著增加延迟，应使用任务路由而不是所有请求固定高分辨率。

## 13.10 Encoder-free 路线的工程收益和代价

### 13.10.1 可能收益

1. 减少独立 encoder 与 connector 的桥接接口。
2. 让图文交错和跨模态生成使用更直接的上下文。
3. 有机会复用部分主干、cache 和训练基础设施，但实际共享范围取决于架构和运行时实现。
4. 统一任务 schema，便于组合输入输出。

### 13.10.2 可能代价

1. 输入序列变长，attention 和 KV 成本上升。
2. 统一主干要同时处理不同统计分布和位置结构。
3. tokenizer、patchifier、codec 和 decoder 仍然需要维护。
4. 模态之间的错误可能互相污染。
5. 细粒度 OCR、时间定位和空间关系仍可能需要专用路径。

“减少模块”不等于“减少责任”。原来由视觉 encoder 承担的分辨率和局部结构问题，可能转移到 tokenizer、主干和路由器。

## 13.11 公开研究路线如何提供证据

下表用公开论文和技术报告说明几种“统一”路线的差异，不把它们当作所有商业产品的内部架构：

| 路线 | 公开资料能支持的结论 | 不能直接推出 |
|---|---|---|
| [Chameleon（论文 v2）](https://arxiv.org/abs/2405.09818v2) | early-fusion、token-based 的混合模态模型可处理图文交错输入与输出 | 所有商业 omni 模型采用相同结构；token-based 就表示没有 tokenizer |
| [Unified-IO 2（论文 v1）](https://arxiv.org/abs/2312.17172v1) | 图像、文本、音频、动作等被整理为统一任务表示，并由一个 encoder-decoder Transformer 处理 | 统一任务表示意味着取消各模态的 tokenizer；生产延迟和安全行为 |
| [Emu3（论文 v1）](https://arxiv.org/abs/2409.18869v1) | 图像、文本和视频经离散化后，在单一 Transformer 中用 next-token 路线建模 | 离散表示无损，或所有媒体的 token 数与成本都和文本相同 |
| [Show-o（论文 v7）](https://arxiv.org/abs/2408.12528v7) | 一个 Transformer 结合自回归建模与离散扩散以适配不同视觉任务 | 一个目标适用于所有模态，或架构没有模态专属前后处理 |
| [Transfusion（论文 v1）](https://arxiv.org/abs/2408.11039v1) | 文本 next-token loss 与图像 diffusion 可在共享模型中协同；论文还加入模态专属编码/解码层 | 共享 Transformer 等于没有模态专属模块，或音频、视频和工具能力自动成立 |
| [ImageBind（论文 v2）](https://arxiv.org/abs/2305.05665v2) | 为六种模态分别使用 encoder 和线性投影头，再把归一化向量对齐到共同嵌入空间 | 共享嵌入空间等于统一 token 序列、共享自回归主干或无专用 encoder |
| [Qwen3.5-Omni（技术报告 v2）](https://arxiv.org/abs/2604.15804v2) | 报告列出 AuT 音频 encoder、SigLIP2 视觉 encoder、Thinker/Talker 等组件，展示统一交互系统仍可保留专用前端 | unified、native 或流式交互等于 encoder-free；报告中的配置代表所有产品版本 |

这些路线并非同一评测下的横向性能对照。模型名右侧的括号固定论文版本；描述限定于该版本公开的方法，不比较作者报告的成绩。表格也给出阅读顺序：先看 token 如何形成，再看主干和目标共享到哪一层，最后检查哪些前端和输出部件仍然存在。若产品只公开“支持图片输入”，教材最多写接口能力；若技术报告只公开部分数据流，就不要补写未披露的 tokenizer、位置、mask 或处理器行为。

## 13.12 部署协议：处理器也是模型的一部分

统一模型上线时，processor 负责把图片、音频、视频、文件和文本序列化。它决定 resize、颜色空间、帧率、采样点、special token、媒体 placeholder、位置和 mask。不同 revision 的 processor 可能产生不同序列，即使输入媒体字节完全相同。

一个可回放 manifest 至少保存：

```json
{
  "media_digest": "sha256:...",
  "media_kind": "document_page",
  "shape_or_rate": "2048x1536",
  "processor_revision": "processor-r7",
  "media_tokens": 3072,
  "position_schema": "2d_patch_v2",
  "mask_schema": "mixed_modal_v1",
  "model_revision": "model-r12",
  "tenant": "tenant-a",
  "output_events": ["text_delta", "citation"]
}
```

只保存图片 URL 或音频路径无法证明线上使用了同一份预处理。缓存 key 还要包含租户和权限，避免私有上下文跨请求复用。

处理器 manifest 不宜只写一个名字。回放时还要能找回实际预处理结果的形状、颜色/采样规范、裁剪或帧选择、token 数、位置与 mask schema，以及处理器和模型各自的不可变 revision。原始媒体 digest 用于确认输入版本，租户和权限字段用于隔离缓存；这些字段不应把原始用户媒体或凭据复制进审计日志。每次部署变更都可先对同一组 golden request 比较实际序列长度、媒体边界和答案支持证据，再判断差异来自权重还是 processor。

## 13.13 评估：接口声明、可用能力和有效证据

评估应分为五层：

1. 接口：能否接收目标媒体、格式和流式事件。
2. 解析：token、区域、时间和 metadata 是否正确。
3. 对齐：问题是否读取了相关证据。
4. 推理：数值、关系和时间结论是否正确。
5. 生成/执行：输出质量、格式、权限和副作用是否正确。

同一张表格截图可以同时测 OCR CER、行列定位、数值计算、引用支持和最终回答。只测 VQA accuracy，可能把 OCR 正确但引用错误的情况隐藏掉。

例如，用 50 份合成合同走完一条固定数据链：46 份含有可读的目标字段，43 份在这 46 份里定位到正确区域，40 份在这 43 份里算对请求值，40 个正确答案中有 38 个能引用到支持它的原文。对应的 OCR 覆盖是 46/50=0.92，区域定位的条件准确率是 43/46≈0.935，数值计算的条件准确率是 40/43≈0.930，引用支持率是 38/40=0.95；全链路同时正确的比例是 38/50=0.76。若只按所有合同计算正确数值，准确率为 40/50=0.80，它不要求定位和引用也同时通过。它们回答不同问题，不能直接平均成一个“多模态能力分”。若 OCR 覆盖高但区域定位下降，排查位置绑定；若答案准确但引用支持低，排查证据对齐；若只有高分辨率恢复定位，则回看采样和预算。

这组分母要在评测前固定：若把没有可读字段的 4 份合同从总数中直接剔除，答案准确率就会掩盖输入解析失败。拒答场景还需分别报告可回答子集的答案质量、不可回答子集的拒答召回和整体任务成功，不能把拒答样本当成普通错误或从统计中消失。通用不确定性、显著性和人工评分规约沿用第十一章，本节重点是把指标映射回本章的表示与数据路径。

可以用 20 个含有明确金额冲突的合成表格请求演示一个捷径指标：把正确 OCR 中的金额替换成相反数值后，若模型有 15 次跟随错误 OCR，定义“错误转写跟随率”为 15/20=0.75。这个 75% 只描述该组反事实样本，不等于一般能力或因果效应；还要对照原图、正确 OCR、只给 OCR 和图像单独输入，并记录每种条件下的分母。改变一个因素并复用同一题目，才能把失败从识别、对齐或推理环节中逐步定位出来。

### 13.13.1 反事实评估

对图表改变颜色、数值或行顺序，对视频打乱帧序或移动关键帧，对音频改变说话人或背景噪声。若输出不随关键证据变化，说明模型没有使用该模态或使用了错误捷径。

反事实最好按同一题成对保存，而不是拿两批不同题目的分数相减。对每对样本记录原始图像、被修改的 OCR 字段、预期答案方向、实际答案和模型是否引用该区域。只有问题语义、图像布局、提示、采样和解码都固定时，答案向改写金额翻转才是“模型可能追随 OCR”的证据；仍需检查 OCR 位置变化和文字可见性等伴随因素。结果应按样本、字段类型与媒体质量分层，避免少数极端样本主导总体比例。

同一配对原则也适用于音视频：把语音中的关键实体或否定词替换成最小对照，检查回答是否跟随变化；对视频只交换两帧次序或遮掉一个关键时间片，观察时间判断是否相应翻转。每项改动都要写明预期变化，否则只能说输出不稳定，不能说模型忽略了哪一类证据。

### 13.13.2 预算和质量同时记录

每个评估样本记录：

- 文本、图像、音频、视频和特殊 token 数。
- 处理时长、prefill、decode 和 cache。
- 关键字段准确率、区域/时间支持率。
- 证据引用和不确定性。
- 失败类型、显存和单位任务成本。

只有同时记录质量和成本，才能判断统一主干是否带来实际收益。

## 13.14 安全：外部媒体是证据，不是系统指令

OWASP GenAI LLM Top 10 2026 将图片、音频、视频等多模态内容列为 LLM01:2026 Prompt Injection 的输入来源，并建议采用纵深防御；它明确指出单一控制不足以应对这类风险。[官方条目](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md)提供风险分类和缓解指导，不是对某个模型或单项控制措施的效果实测。

信任边界应当作为结构化的来源元数据一路传递，而不是只在 system prompt 中写一句“忽略媒体指令”。OCR 服务如果把图片中的文字提取出来，就要保留它来自用户媒体的标签；序列化器插入这些 span 时不能把角色改成 system 或 tool。模型可以把可疑句子作为被引用的内容解释，但访问文件、发送消息、下单或执行代码仍由独立执行器根据用户授权、参数白名单、资源范围和确认状态决定。评估也应把“识别出恶意文字”“没有服从它”和“没有产生外部副作用”分开计数；一个环节通过不能代替其他环节。

图片、PDF、网页截图、音频和视频都可以藏有自然语言指令。`encoder-free` 并不会自动解决 prompt injection，反而可能让更多外部 token 直接进入共享主干。系统应在 schema 中保留信任标签：

```text
system policy      -> trusted control
user request       -> user instruction
media text         -> untrusted evidence
retrieval result   -> external evidence
tool result        -> observation
model plan         -> proposed action
executor           -> permission check
```

模型可以解释媒体中的恶意文字，但不能让它覆盖系统规则、改变权限或直接触发副作用。工具执行器应独立检查用户授权、参数、资源范围和确认状态。

## 13.15 失败模式：从“模型不聪明”回到可定位根因

### 13.15.1 静默缩放或截断

图片被缩小、视频被抽稀、音频被截断，却没有在 trace 中记录。模型看不到关键证据，最终表现像推理错误。

### 13.15.2 位置和 mask 错误

文本问题与图片区域错配，未来视频帧可见，或多张图片边界丢失。输出仍可能流畅，必须通过 golden request 和反事实测试发现。

### 13.15.3 模态捷径

模型过度依赖 OCR、caption 或字幕，忽略原始图像、音频和视频。添加错误转写、移除转写和提供金标准证据可以定位这种捷径。

### 13.15.4 训练和部署不一致

训练时的模态顺序、processor、special token、位置 schema 和线上不同。权重可以加载，任务质量却会下降。版本字段必须成为回放和缓存契约的一部分。

工程排障可以从一条错误回答反向回放，而不是先归因于模型推理能力。先核对媒体 digest 和处理器 revision，再对照实际 resize、裁剪、帧采样、时长和 token 账本；确认序列中的媒体边界与位置是否仍指向原证据；随后检查问题 query 是否能看到目标 token、答案是否按正确方向生成；最后用正确 OCR、错误 OCR、只给媒体和只给 OCR 的成对消融定位捷径。若离线 golden request 正确而线上失败，优先比较服务端 processor、mask schema 和模型 revision；若所有条件都无法指向字段，则把问题归入数据覆盖或模型能力，安排新的证据实验。日志至少要能记录各阶段版本、输入摘要、预处理结果、token 数、失败层和可复现样本 ID。

做这种逐层回放时，一次只改变一个关键条件。若同时换了图像分辨率、OCR 结果和提示，就无法判断哪个改动让答案翻转；若采用随机采样，则还应固定解码参数与随机种子，必要时对同一条件重复运行并报告波动。OCR span 应和坐标、置信度、来源页面绑定，避免只保存识别后的纯文字。失败报告至少给出样本总数、被筛除样本、各层条件分母和预先定义的错误类别；这样能区分“没读到证据”“读到但位置错”“看见证据却推错”和“答案正确但引用无关”。排障结果只支持对应的模型、处理器、输入和解码条件，不能据少数案例概括整类系统。

## 13.16 最小可运行的统一 token 审计

下面的标准库示例检查一份教学 manifest 是否把 token 预算、位置字段、mask 规则、处理器版本和反事实测试计划声明完整。它不实现 encoder-free 模型，也不读取图片、音频或视频；输入尺寸和速率是为演算构造的假设。程序输出的是预算代理和清单一致性，不能证明实际 tokenizer 产生了相同序列，更不能证明模型看见了正确区域或完成了实验。

先手算 token 数，代码会复用同一组输入。图片按每边 14 像素一块切分，336/14=24，所以有 24×24=576 个 patch。音频假设每秒产生 50 个时间帧、每帧有 4 个 codebook token，2.4×50×4=480。视频每 2 帧合成一个时间块，8/2=4 个时间块；每帧空间网格是 224/16=14×14，所以有 4×14×14=784 个视频 token。再加 96 个文本 token 和 12 个边界/角色 token，总数为 96+576+480+784+12=1,948。

| 输入部分 | 假设与计算 | token 数 |
|---|---|---:|
| 文本 | 样本给定 | 96 |
| 图像 | ceil(336/14)×ceil(336/14) | 576 |
| 音频 | ceil(2.4×50)×4 个 codebook | 480 |
| 视频 | ceil(8/2)×ceil(224/16)×ceil(224/16) | 784 |
| 边界和角色 | 样本给定 | 12 |
| 合计 | 五项相加 | 1,948 |

在这个简化布局里，图像网格的位置可以写成 24×24 个 (行, 列) 坐标；音频可用 (时间步, codebook)；视频可用 (时间块, 行, 列)。代码只核对 manifest 声明了这些位置类型，并未逐个建立或检查真实 token 的坐标。它也只核对任务 mask 契约是否被声明完整：本例是媒体问答，要求问题能读取输入媒体、答案按自回归顺序生成、媒体边界保留。它不要求视频问答屏蔽未来帧，因为输入视频的后续帧也是可用证据；若任务改为视频自回归生成，则应另行声明并核对未来帧不可见。代码没有构造 Transformer 的注意力矩阵，不能验证线上 kernel 是否真的照此执行。

把 1,948 个位置两两组合，得到 1,948²=3,794,704 个 score-cell 代理。这只是一个序列、一个 attention head、一个 layer 的稠密矩阵尺寸示意；多头、多层、batch、padding、窗口化/稀疏注意力及后端实现都会改变实际工作量。该数不能当成 FLOPs、显存、延迟或服务成本。第十四章会把请求级采样、缓存和真实部署测量放回完整预算。

代码逐项核对输入 schema，特别拒绝 bool 伪装成整数、负尺寸、NaN/无穷时长和错误的 media trust 类型。超大整数时长不经过浮点有限性转换，仍可算出一个很大的 token 数并由 context_fits 标成超预算。positions_complete、mask_rules_declared 和 counterfactual_plan_complete 只回答配置字段是否完整；反事实列表里列出了原媒体、正确/错误转写和打乱视频顺序四种条件，但程序不会运行模型或证明结果随输入改变。模型卡或代码中的结构声明仍需回到固定 revision、真实 processor 输出、attention mask 与模型实验分别核验。

运行方式是将下方代码保存为 Python 文件并直接执行；它只依赖 Python 标准库。首先查看 token_counts 和 total_tokens 是否与手算一致，再确认 checks 每一项的具体含义。改变 image_patch、audio_frames_per_second、audio_codebooks、video_temporal_patch 或 context_limit，会分别改变相应 token 项或预算结论。把图像 patch 从 14 改为 28 时，图像 token 从 24²=576 降到 12²=144，序列少 432 个 token；在同样的全注意力代理下，总长度从 1,948 变为 1,516，score-cell 从 3,794,704 变为 2,298,256。此变化只是本假设下的算术结果，也不说明细节质量保持不变。

> **代码状态：已验证（2026-11-12；Python 3.12.9；仅标准库）。**

```python
import math
from math import ceil


REQUIRED_POSITIONS = {
    "text": "1d",
    "image": "2d",
    "audio": "time+codebook",
    "video": "3d",
}
REQUIRED_MASK_RULES = {
    "question_reads_media",
    "answer_causal",
    "media_boundaries_preserved",
}
REQUIRED_COUNTERFACTUALS = {
    "original_media",
    "correct_transcript",
    "incorrect_transcript",
    "shuffled_video_order",
}


def ceil_div(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("ceil_div arguments must be integers")
    if a <= 0 or b <= 0:
        raise ValueError("ceil_div arguments must be positive")
    return (a + b - 1) // b


def require_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def require_positive_int(value, name):
    value = require_nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def require_exact_keys(value, expected, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(f"{name} has an invalid schema")
    return value


def validate_sample_and_config(sample, config):
    sample_keys = {"text_tokens", "images", "audios", "videos", "media_trust"}
    config_keys = {
        "image_patch", "video_patch", "video_temporal_patch",
        "audio_frames_per_second", "audio_codebooks", "special_tokens",
        "context_limit", "processor_revision", "position_schema",
        "mask_schema", "counterfactual_plan",
    }
    require_exact_keys(sample, sample_keys, "sample")
    require_exact_keys(config, config_keys, "config")

    require_nonnegative_int(sample["text_tokens"], "text_tokens")
    if not isinstance(sample["images"], list) or not isinstance(sample["audios"], list) or not isinstance(sample["videos"], list):
        raise TypeError("media collections must be lists")
    if not isinstance(sample["media_trust"], str):
        raise TypeError("media_trust must be a string")

    require_positive_int(config["image_patch"], "image_patch")
    require_positive_int(config["video_patch"], "video_patch")
    require_positive_int(config["video_temporal_patch"], "video_temporal_patch")
    require_positive_int(config["audio_frames_per_second"], "audio_frames_per_second")
    require_positive_int(config["audio_codebooks"], "audio_codebooks")
    require_nonnegative_int(config["special_tokens"], "special_tokens")
    require_positive_int(config["context_limit"], "context_limit")
    if not isinstance(config["processor_revision"], str):
        raise TypeError("processor_revision must be a string")

    position_schema = require_exact_keys(
        config["position_schema"], set(REQUIRED_POSITIONS), "position_schema"
    )
    if any(not isinstance(value, str) or not value.strip() for value in position_schema.values()):
        raise ValueError("position schema values must be non-empty strings")

    mask_schema = require_exact_keys(
        config["mask_schema"], REQUIRED_MASK_RULES, "mask_schema"
    )
    if any(type(value) is not bool for value in mask_schema.values()):
        raise TypeError("mask schema values must be booleans")

    plan = config["counterfactual_plan"]
    if not isinstance(plan, list) or any(not isinstance(item, str) for item in plan):
        raise TypeError("counterfactual_plan must be a list of strings")
    if len(plan) != len(set(plan)):
        raise ValueError("counterfactual_plan must not contain duplicates")

    for index, image in enumerate(sample["images"]):
        require_exact_keys(image, {"height", "width"}, f"images[{index}]")
        require_positive_int(image["height"], f"images[{index}].height")
        require_positive_int(image["width"], f"images[{index}].width")

    for index, audio in enumerate(sample["audios"]):
        require_exact_keys(audio, {"seconds"}, f"audios[{index}]")
        require_finite_nonnegative(audio["seconds"], f"audios[{index}].seconds")

    for index, video in enumerate(sample["videos"]):
        require_exact_keys(video, {"frames", "height", "width"}, f"videos[{index}]")
        require_positive_int(video["frames"], f"videos[{index}].frames")
        require_positive_int(video["height"], f"videos[{index}].height")
        require_positive_int(video["width"], f"videos[{index}].width")


def ceil_audio_frames(seconds, frames_per_second):
    if isinstance(seconds, int):
        return seconds * frames_per_second
    product = seconds * frames_per_second
    if not math.isfinite(product):
        raise ValueError("audio duration is too large for floating-point arithmetic")
    return ceil(product)


def build_audit(sample, config):
    if not isinstance(sample, dict) or not isinstance(config, dict):
        raise TypeError("sample and config must be mappings")
    validate_sample_and_config(sample, config)
    if not sample["text_tokens"] and not sample["images"] and not sample["audios"] and not sample["videos"]:
        raise ValueError("request must contain text or media")

    image_tokens = sum(
        ceil_div(image["height"], config["image_patch"])
        * ceil_div(image["width"], config["image_patch"])
        for image in sample["images"]
    )
    audio_tokens = sum(
        ceil_audio_frames(audio["seconds"], config["audio_frames_per_second"])
        * config["audio_codebooks"]
        for audio in sample["audios"]
    )
    video_tokens = sum(
        ceil_div(video["frames"], config["video_temporal_patch"])
        * ceil_div(video["height"], config["video_patch"])
        * ceil_div(video["width"], config["video_patch"])
        for video in sample["videos"]
    )
    token_counts = {
        "text": sample["text_tokens"],
        "image": image_tokens,
        "audio": audio_tokens,
        "video": video_tokens,
        "special": config["special_tokens"],
    }
    total_tokens = sum(token_counts.values())
    if total_tokens == 0:
        raise ValueError("request must contain at least one token")

    checks = {
        "context_fits": total_tokens <= config["context_limit"],
        "positions_complete": config["position_schema"] == REQUIRED_POSITIONS,
        "mask_rules_declared": all(config["mask_schema"].values()),
        "counterfactual_plan_complete": REQUIRED_COUNTERFACTUALS.issubset(
            set(config["counterfactual_plan"])
        ),
        "processor_pinned": bool(config["processor_revision"].strip()),
        "trust_boundary_declared": sample["media_trust"] == "untrusted_evidence",
    }
    return {
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "attention_cells_proxy": total_tokens * total_tokens,
        "position_schema": config["position_schema"],
        "mask_schema": config["mask_schema"],
        "counterfactual_plan": config["counterfactual_plan"],
        "checks": checks,
        "manifest_complete": all(checks.values()),
    }


sample = {
    "text_tokens": 96,
    "images": [{"height": 336, "width": 336}],
    "audios": [{"seconds": 2.4}],
    "videos": [{"frames": 8, "height": 224, "width": 224}],
    "media_trust": "untrusted_evidence",
}
config = {
    "image_patch": 14,
    "video_patch": 16,
    "video_temporal_patch": 2,
    "audio_frames_per_second": 50,
    "audio_codebooks": 4,
    "special_tokens": 12,
    "context_limit": 4096,
    "processor_revision": "processor-r7",
    "position_schema": {
        "text": "1d",
        "image": "2d",
        "audio": "time+codebook",
        "video": "3d",
    },
    "mask_schema": {
        "question_reads_media": True,
        "answer_causal": True,
        "media_boundaries_preserved": True,
    },
    "counterfactual_plan": [
        "original_media",
        "correct_transcript",
        "incorrect_transcript",
        "shuffled_video_order",
    ],
}

report = build_audit(sample, config)
for key, value in report.items():
    print(f"{key}={value}")
```

参考输出为：

```text
token_counts={'text': 96, 'image': 576, 'audio': 480, 'video': 784, 'special': 12}
total_tokens=1948
attention_cells_proxy=3794704
position_schema={'text': '1d', 'image': '2d', 'audio': 'time+codebook', 'video': '3d'}
mask_schema={'question_reads_media': True, 'answer_causal': True, 'media_boundaries_preserved': True}
counterfactual_plan=['original_media', 'correct_transcript', 'incorrect_transcript', 'shuffled_video_order']
checks={'context_fits': True, 'positions_complete': True, 'mask_rules_declared': True, 'counterfactual_plan_complete': True, 'processor_pinned': True, 'trust_boundary_declared': True}
manifest_complete=True
```

通过手算与代码对照，可以检查尺寸到 token 数、token 数到全注意力代理的算术是否一致。边界输入的意义是确保清单校验不会把配置错误误判为通过：布尔值不能冒充 token 整数，NaN/无穷秒数不能进入预算，非字符串 trust 值应被拒绝；若处理器 revision 为空或媒体信任标签不是 untrusted_evidence，对应检查应返回 False；缺少位置字段或重复反事实条件应明确报错或使对应检查失败。即便所有 checks 都为 True，也只表示这份教学 manifest 自洽；还要在真实模型中记录序列化后的 token、位置、mask 张量及模型/processor revision，才能验证实际数据路径。

如果 positions_complete 为 False，正确做法是检查实际模型的坐标约定并更新 schema，而不是把模型强行改写成这里的一维/二维/三维模板；如果 mask_rules_declared 为 False，则要先查明任务是媒体理解、因果生成还是双向修复，再定义可见关系。counterfactual_plan_complete 为 True 只说明计划列齐，实验结果必须由外部评测器保存。这个区分让 demo 的绿色状态保持在配置门槛，不会被误读为模型质量或安全验收通过。

## 13.17 练习：把术语边界落实到实验

### 练习一：拆分一个 encoder-free 声明

某模型卡写“无需视觉 encoder 的统一多模态模型”。请把它拆成 tokenizer、patchifier、主干、输出 decoder、训练目标和部署协议六个问题，列出每个问题需要的证据。

### 练习二：位置设计

为图文交错、视频生成和音频 codec 序列分别设计位置字段，说明为什么一个普通一维位置不足以表示它们。

### 练习三：mask 设计

设计一个图像理解 mask 和一个视频自回归生成 mask，标出问题 token、视觉 token、历史帧和未来帧之间的可见关系。

### 练习四：反事实模态消融

为表格截图设计原图、正确 OCR、错误 OCR、只给 OCR 四个条件，并说明每种结果对应哪一种错误归因。

### 练习五：token 预算

计算 `336 × 336` 图像、2.4 秒音频和 8 帧视频的 token 数，并说明改变 patch、帧率和 codebook 数分别影响什么。

### 练习六：统一目标冲突

设计文本理解、图像生成和安全拒答三个 loss，说明按样本平均和按 token 平均可能导致的差异。

### 练习七：部署回放

设计一个 golden manifest，至少包含媒体 hash、processor revision、位置 schema、mask、token 数、模型 revision、租户和输出事件。

### 练习八：安全边界

构造一张包含恶意文字的图片，要求模型总结它的内容但不执行它的指令。说明模型、路由器和工具执行器分别承担什么责任。

## 13.18 资料与证据边界

本章主要引用以下公开研究：

1. Chameleon Team, [Chameleon: Mixed-Modal Early-Fusion Foundation Models（arXiv v2）](https://arxiv.org/abs/2405.09818v2)。
2. Lu et al., [Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action（arXiv v1）](https://arxiv.org/abs/2312.17172v1)。
3. Emu3 Team, [Emu3: Next-Token Prediction is All You Need（arXiv v1）](https://arxiv.org/abs/2409.18869v1)。
4. Xie et al., [Show-o: One Single Transformer to Unify Multimodal Understanding and Generation（arXiv v7）](https://arxiv.org/abs/2408.12528v7)。
5. Zhou et al., [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model（arXiv v1）](https://arxiv.org/abs/2408.11039v1)。
6. Girdhar et al., [ImageBind: One Embedding Space To Bind Them All（arXiv v2）](https://arxiv.org/abs/2305.05665v2)。
7. Qwen Team, [Qwen3.5-Omni Technical Report（arXiv v2）](https://arxiv.org/abs/2604.15804v2)。
8. Unified multimodal 系统中的视觉、文档、视频、语音评估和安全资料，见本册[第十一章的资料与证据边界](11-多模态评估与安全.md)。

这些论文支持各自公开的 token 化、融合或训练机制，不支持对未公开商业模型内部结构的推断。“Encoder-free”在不同资料中可能有不同范围，正文应明确说明它减少的是哪一类独立 encoder，以及哪些 tokenizer、patchifier、codec、projection 和 decoder 仍然存在。
