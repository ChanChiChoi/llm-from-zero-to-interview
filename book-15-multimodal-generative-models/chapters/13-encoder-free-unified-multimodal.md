# 第十三章：Encoder-Free 统一多模态——先把术语说清楚，再讨论架构

“Encoder-free unified multimodal”听起来像一个非常明确的技术结论：模型没有视觉编码器，也没有音频编码器，所有模态直接进入同一个 Transformer。但在真实论文、模型卡和产品宣传中，`encoder-free` 的含义并不总是一致。有的作者指的是没有独立的视觉语言 encoder；有的指的是图像被离散 tokenizer 直接变成 token；有的只是强调理解和生成共享主干；还有的系统仍然使用 patchifier、codec、embedding、projection 或模态适配器。

因此本章先解决词义，再讨论结构。我们会把“前端编码器”“模态 tokenizer”“统一主干”“输出 decoder”拆开，说明它们各自承担什么责任；随后讨论统一 token 流的序列化、位置和 mask、训练目标、信息压缩、评估、部署协议和安全边界。最后用一个可运行的审计示例，检查不同模态 token、位置、预算和反事实实验是否自洽。

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

不同模态被编码到可以互相访问的序列或 latent 空间：

~~~math
z_m=E_m(x_m)\in\mathbb{R}^{N_m\times d}
~~~

这里的 `d` 统一了 hidden size，但不等于每种模态的信息语义已经完全对齐。图像 token 仍然需要二维位置，视频 token 仍然需要时间索引，音频 token 仍然有采样率和 codebook 层级。

这个抽象至少要求 `N_m` 是非负整数、`d>0`，并且每种 `E_m` 的输出 dtype、设备和特殊 token 约定可以被主干接受。`N_m` 相同也不意味着 token 可以互换：一个图像 token 的坐标、一个音频 token 的时间和一个工具结果的权限来源，仍然需要元数据和 mask 共同解释。

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
Z=\operatorname{Interleave}(Z_t,Z_i,Z_a,Z_v,Z_s)
~~~

`Z_s` 是模态边界、角色、时间戳、坐标和版本等特殊或元数据元素。统一主干计算：

~~~math
H=\operatorname{Transformer}(Z;M,\Pi)
~~~

`M` 是 attention visibility 或 causal mask，`Pi` 是位置结构。这个抽象表达的是共享信息路由，不是所有模态都使用相同 tokenizer、相同位置编码或相同输出头。

若统一序列为空，主干是否允许空上下文需要单独定义；若 `Z_s` 没有记录媒体边界、角色或版本，交错顺序本身不足以恢复来源。对同一个请求，序列化器还应保证 token、位置和 mask 使用同一索引，否则“看见了正确 token”仍可能等价于读取了错误区域。

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

## 13.6 Token 化路线：连续、离散和混合

### 13.6.1 连续表示

视觉或音频编码器输出浮点 hidden states：

~~~math
H_m\in\mathbb{R}^{N_m\times d_m}
~~~

连续表示保留较丰富的几何信息，适合理解和 cross-attention，但生成端需要专用 decoder、扩散过程或声码器。

### 13.6.2 离散表示

离散 tokenizer 把输入映射成有限词表中的整数：

~~~math
z_m=Q(E_m(x_m)),
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
3. 共享部分主干、cache 和训练基础设施。
4. 统一任务 schema，便于组合输入输出。

### 13.10.2 可能代价

1. 输入序列变长，attention 和 KV 成本上升。
2. 统一主干要同时处理不同统计分布和位置结构。
3. tokenizer、patchifier、codec 和 decoder 仍然需要维护。
4. 模态之间的错误可能互相污染。
5. 细粒度 OCR、时间定位和空间关系仍可能需要专用路径。

“减少模块”不等于“减少责任”。原来由视觉 encoder 承担的分辨率和局部结构问题，可能转移到 tokenizer、主干和路由器。

## 13.11 公开研究路线如何提供证据

本节只使用公开论文能够确认的机制，不把它们当作所有商业产品的内部架构：

| 路线 | 公开资料能支持的结论 | 不能直接推出 |
|---|---|---|
| Chameleon | mixed-modal early-fusion、token-based 的图文混合建模 | 所有商业 omni 模型都采用相同实现 |
| Unified-IO 2 | 将多类输入输出 token 化，用统一 encoder-decoder 处理 | 生产系统的实时延迟和安全行为 |
| Emu3 | 图像、文本和视频离散化后使用 next-token 路线 | 所有媒体的细节和成本都与文本相同 |
| Show-o | 共享 Transformer 中结合自回归和离散扩散 | 一个目标适用于所有模态 |
| Transfusion | 文本 next-token 与图像 diffusion 在共享模型中结合 | 音频、视频和工具能力自动成立 |

如果产品只公开“支持图片输入”，教材最多写接口能力；如果技术报告公开了统一 token 主干，才能写对应的架构事实；如果 tokenizer 和内部层未公开，就保留限定语。

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

## 13.13 评估：接口声明、可用能力和有效证据

评估应分为五层：

1. 接口：能否接收目标媒体、格式和流式事件。
2. 解析：token、区域、时间和 metadata 是否正确。
3. 对齐：问题是否读取了相关证据。
4. 推理：数值、关系和时间结论是否正确。
5. 生成/执行：输出质量、格式、权限和副作用是否正确。

同一张表格截图可以同时测 OCR CER、行列定位、数值计算、引用支持和最终回答。只测 VQA accuracy，可能把 OCR 正确但引用错误的情况隐藏掉。

### 13.13.1 反事实评估

对图表改变颜色、数值或行顺序，对视频打乱帧序或移动关键帧，对音频改变说话人或背景噪声。若输出不随关键证据变化，说明模型没有使用该模态或使用了错误捷径。

### 13.13.2 预算和质量同时记录

每个评估样本记录：

- 文本、图像、音频、视频和特殊 token 数。
- 处理时长、prefill、decode 和 cache。
- 关键字段准确率、区域/时间支持率。
- 证据引用和不确定性。
- 失败类型、显存和单位任务成本。

只有同时记录质量和成本，才能判断统一主干是否带来实际收益。

## 13.14 安全：外部媒体是证据，不是系统指令

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

## 13.16 最小可运行的统一 token 审计

下面的示例不实现真实 encoder-free 模型，只审计一条混合请求的 token、模态位置、预算和反事实条件。它故意把 `encoder_free` 解释成“没有独立视觉语言 encoder 的抽象路线”，并保留 patchifier、audio codec 和 processor 字段，避免把术语误解成没有任何前端处理。

```python
import math
from math import ceil


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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def validate_sample_and_config(sample, config):
    required_sample = {"text_tokens", "images", "audios", "videos", "media_trust"}
    required_config = {
        "image_patch", "video_patch", "video_temporal_patch", "audio_rate",
        "audio_codebooks", "special_tokens", "context_limit", "processor_revision",
    }
    if set(sample) != required_sample or set(config) != required_config:
        raise ValueError("sample or config has an invalid schema")
    require_nonnegative_int(sample["text_tokens"], "text_tokens")
    if not isinstance(sample["images"], list) or not isinstance(sample["audios"], list) or not isinstance(sample["videos"], list):
        raise TypeError("media collections must be lists")
    if sample["media_trust"] not in {"untrusted_evidence"}:
        raise ValueError("media_trust must be untrusted_evidence")
    require_positive_int(config["image_patch"], "image_patch")
    require_positive_int(config["video_patch"], "video_patch")
    require_positive_int(config["video_temporal_patch"], "video_temporal_patch")
    require_positive_int(config["audio_rate"], "audio_rate")
    require_positive_int(config["audio_codebooks"], "audio_codebooks")
    require_nonnegative_int(config["special_tokens"], "special_tokens")
    require_positive_int(config["context_limit"], "context_limit")
    if not isinstance(config["processor_revision"], str) or not config["processor_revision"]:
        raise ValueError("processor_revision must be non-empty")
    for index, image in enumerate(sample["images"]):
        if set(image) != {"height", "width"}:
            raise ValueError(f"images[{index}] has an invalid schema")
        require_positive_int(image["height"], f"images[{index}].height")
        require_positive_int(image["width"], f"images[{index}].width")
    for index, audio in enumerate(sample["audios"]):
        if set(audio) != {"seconds"}:
            raise ValueError(f"audios[{index}] has an invalid schema")
        require_finite_nonnegative(audio["seconds"], f"audios[{index}].seconds")
    for index, video in enumerate(sample["videos"]):
        if set(video) != {"frames", "height", "width"}:
            raise ValueError(f"videos[{index}] has an invalid schema")
        require_positive_int(video["frames"], f"videos[{index}].frames")
        require_positive_int(video["height"], f"videos[{index}].height")
        require_positive_int(video["width"], f"videos[{index}].width")


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
        ceil(audio["seconds"] * config["audio_rate"])
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

    positions = {
        "text": "1d",
        "image": "2d",
        "audio": "time+codebook",
        "video": "3d",
    }
    counterfactuals = {
        "original_media": True,
        "correct_transcript": True,
        "incorrect_transcript": True,
        "shuffled_video_order": True,
    }
    checks = {
        "context_fits": total_tokens <= config["context_limit"],
        "positions_declared": set(positions) == {"text", "image", "audio", "video"},
        "counterfactuals_declared": all(counterfactuals.values()),
        "processor_pinned": bool(config["processor_revision"]),
        "trust_boundary_declared": sample["media_trust"] == "untrusted_evidence",
    }
    return {
        "route": {
            "encoder_free_claim": True,
            "frontends": ["image_patchifier", "audio_codec", "video_sampler"],
            "shared_backbone": "mixed_modal_transformer",
            "output_decoder": ["text_decoder", "tool_executor"],
        },
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "attention_cells_proxy": total_tokens * total_tokens,
        "positions": positions,
        "counterfactuals": counterfactuals,
        "checks": checks,
        "audit_consistent": all(checks.values()),
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
    "audio_rate": 50,
    "audio_codebooks": 4,
    "special_tokens": 12,
    "context_limit": 4096,
    "processor_revision": "processor-r7",
}

report = build_audit(sample, config)
for key, value in report.items():
    print(f"{key}={value}")
```

参考输出为：

```text
route={'encoder_free_claim': True, 'frontends': ['image_patchifier', 'audio_codec', 'video_sampler'], 'shared_backbone': 'mixed_modal_transformer', 'output_decoder': ['text_decoder', 'tool_executor']}
token_counts={'text': 96, 'image': 576, 'audio': 480, 'video': 784, 'special': 12}
total_tokens=1948
attention_cells_proxy=3794704
positions={'text': '1d', 'image': '2d', 'audio': 'time+codebook', 'video': '3d'}
counterfactuals={'original_media': True, 'correct_transcript': True, 'incorrect_transcript': True, 'shuffled_video_order': True}
checks={'context_fits': True, 'positions_declared': True, 'counterfactuals_declared': True, 'processor_pinned': True, 'trust_boundary_declared': True}
audit_consistent=True
```

`encoder_free_claim=True` 只表示这条教学路线的建模假设，不能证明任何真实模型没有视觉 encoder。`frontends` 列表特意保留了 patchifier、codec 和 sampler，说明减少独立 encoder 不等于删除所有模态处理。

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

1. Chameleon: Mixed-Modal Early-Fusion Foundation Models，https://arxiv.org/abs/2405.09818
2. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action，https://arxiv.org/abs/2312.17172
3. Emu3: Next-Token Prediction is All You Need，https://arxiv.org/abs/2409.18869
4. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation，https://arxiv.org/abs/2408.12528
5. Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model，https://arxiv.org/abs/2408.11039
6. ImageBind: One Embedding Space To Bind Them All，https://arxiv.org/abs/2305.05665
7. Unified multimodal 系统中的视觉、文档、视频、语音评估和安全资料，见本册第 11 章第 11.20 节。

这些论文支持各自公开的 token 化、融合或训练机制，不支持对未公开商业模型内部结构的推断。“Encoder-free”在不同资料中可能有不同范围，正文应明确说明它减少的是哪一类独立 encoder，以及哪些 tokenizer、patchifier、codec、projection 和 decoder 仍然存在。
