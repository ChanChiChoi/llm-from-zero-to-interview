# 第三章：Vision Encoder

多模态模型看到的不是图片文件，而是图片经过视觉编码器转换后的表示。原始输入可能是一个形状为 [B, 3, H, W] 的像素张量；经过 vision encoder 后，它可能变成一个全局向量，也可能变成 [B, N, d_v] 的视觉 token 序列。后续的 CLIP 相似度、VLM projector、cross-attention 或语言模型，都建立在这个转换结果上。

因此，vision encoder 不是一个可以随意替换的“前处理模块”。它决定了几个关键问题：

1. 图像中的空间结构是否仍然可定位。
2. 小文字和小物体是否在采样时保留下来。
3. 全局语义与局部细节如何分配表示容量。
4. 视觉 token 有多少，以及它们会占用多少上下文和显存。
5. 输出维度、token 顺序和位置坐标能否与下游模块匹配。
6. 预训练目标鼓励模型学习全局对齐、局部重建还是类别判别。

一个模型在低分辨率自然图片上检索效果很好，不代表它能读懂扫描合同；一个视觉塔输出了很多 token，也不代表这些 token 保留了正确的坐标关系。学习本章时，要始终把表示能力、输入契约和计算成本放在同一张图上。

本章沿着下面的数据流展开：

~~~text
像素与预处理
    -> 局部算子或 patch embedding
    -> 空间位置与视觉主干
    -> 全局向量、patch tokens 或多层特征
    -> projector / pooling / cross-attention
    -> CLIP 检索、VLM 问答或结构化视觉任务
~~~

## 0. 研究对象、资料层级与边界

### 0.1 本章讨论的核心问题

本章讨论视觉编码器在多模态系统中的表示与工程责任：

1. CNN 如何利用局部感受野、权重共享和残差连接。
2. ViT 如何把图像切成 patch，并把 patch 变成 Transformer token。
3. 无 padding、显式 padding、resize 和 crop 如何改变网格大小。
4. patch embedding 的权重形状、位置编码和 token 顺序。
5. CLS token、平均池化、attention pooling 和 patch tokens 的差异。
6. 多层视觉特征、层间语义和局部空间信息。
7. 分辨率、patch size、token 数、attention 成本和显存的关系。
8. CLIP、SigLIP、MAE 等预训练目标对视觉表示的影响。
9. 视觉塔与 projector、LLM hidden size 和上下文预算的接口。
10. OCR、图表、小目标和空间关系为什么需要额外设计。

检测、分割、专用 OCR 模型、视频时序编码和完整 VLM 对话训练会在其他章节展开。本章会在边界处说明它们需要视觉编码器提供什么，但不会用一张架构图替代专门论证。

### 0.2 资料能证明什么

本章使用三类资料：

| 资料 | 可以支持 | 不应直接推出 |
| --- | --- | --- |
| ResNet、ViT、CLIP、SigLIP、MAE 等原始论文 | 论文提出的结构、目标和实验结果 | 任意产品版本的全部实现细节 |
| 官方代码和模型文档 | 特定实现的预处理、权重和张量接口 | 所有视觉塔都采用同样的 pooling 和位置编码 |
| 本章的教学代码 | 形状、算术和成本公式是否自洽 | 真实图片上的识别、OCR 或安全能力 |

例如，ViT 论文支持“patch 序列可以由 Transformer 处理”这一机制命题；OpenAI CLIP 的公开代码支持某个版本使用卷积 patch embedding、class embedding 和位置 embedding 的实现事实；它们不等于对一个新模型的私有视觉塔做了证明。

### 0.3 小白先建立一个直觉

可以把图片想成一张地图：

1. 像素是地图上的原始测量。
2. patch 是把地图切成小方格。
3. patch embedding 把每个方格翻译成向量。
4. 位置编码告诉模型每个方格在地图哪里。
5. Transformer 或 CNN 让局部方格交换信息。
6. 全局向量是整张地图的摘要，patch tokens 是带坐标的局部记录。

摘要适合回答“这是什么类型的图片”，局部记录才更可能支持“左上角表格第三行的数字是什么”。摘要和证据不是同一种输出。

### 0.4 专家要检查哪些契约

工程上至少要保存：

1. 输入通道顺序、颜色空间和数值范围。
2. resize、crop、padding 和插值方式。
3. patch size、网格高度和网格宽度。
4. 是否有 CLS token、其他特殊 token 或位置 token。
5. token 的行列顺序和坐标映射。
6. 视觉 hidden size、投影维度和 dtype。
7. 选用的层、pooling 方式和输出 token 数。
8. 视觉塔的权重版本与预训练目标。

只记录一个模型名称不足以复现视觉输入。相同的权重配上不同的裁剪和归一化，可能产生完全不同的 embedding。

## 1. 从像素到视觉表示

### 1.1 原始张量的形状

一张 RGB 图片可以表示为：

~~~math
X\in\mathbb{R}^{C\times H\times W},\qquad C=3.
~~~

批量输入通常是：

~~~math
X\in\mathbb{R}^{B\times C\times H\times W}.
~~~

这里的 H 和 W 是进入视觉塔的尺寸，不一定是用户上传文件的原始尺寸。预处理可能先保持纵横比缩放，再中心裁剪、边缘填充或切成多个窗口。

视觉编码器的输出常见为三种形式：

1. 全局向量 g，形状为 [B, d_g]。
2. 空间 token Z，形状为 [B, N, d_v]。
3. 多层或多尺度特征，形状可能是 [B, C_l, H_l, W_l] 的集合。

选择哪一种输出，不是代码风格问题，而是任务需求问题。检索通常可以使用 g；图文对话、OCR 和定位通常需要 Z 或多尺度特征。

### 1.2 表示不是原图的无损副本

如果一张 224 x 224 的 RGB 图片使用 patch size 16，单纯的 patch 数是：

~~~math
N=\frac{224}{16}\times\frac{224}{16}=14\times14=196.
~~~

后续每个 token 可以有 768 或 1024 个通道，但 token 数仍然只有 196。模型不可能把所有像素细节以互不压缩的方式逐项传给语言模型。它学习的是对目标任务有用的表示。

这种压缩带来取舍：

1. 低分辨率减少计算，却可能吞掉小字。
2. 大 patch 减少 token，却让一个 token 覆盖更大的空间区域。
3. 全局 pooling 便宜，却可能模糊多个对象的关系。
4. 保留 patch tokens 能支持空间任务，却增加后续 attention 成本。

所以“视觉编码器输出了 embedding”不是完整描述，必须说明 embedding 是否全局、是否保留空间顺序、来自哪一层以及对应多少输入像素。

### 1.3 视觉主干的两种基本路线

CNN 通过局部卷积逐层扩大感受野，天然保留网格结构；ViT 先把图像变成 token，再用 self-attention 进行跨位置交互。两者都可以输出全局和局部特征，但它们的归纳偏置、成本曲线和预训练方式不同。

可以用一个问题区分两条路线：

~~~text
模型在最初阶段如何知道相邻像素有空间关系？
~~~

CNN 把局部卷积和权重共享直接写进结构；ViT 把 patch 作为序列元素，再用位置编码和 attention 学习或利用空间关系。大规模数据可以弥补一部分 ViT 的先验不足，但不代表位置和分辨率契约可以省略。

## 2. CNN：局部归纳偏置与层级特征

### 2.1 卷积的计算

对输入特征图 X 和卷积核 K，一个二维卷积输出可以抽象为：

~~~math
Y_{b,o,h,w}
=\left(
\sum_{c=1}^{C_{\mathrm{in}}}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
K_{o,c,u,v}\,
X_{b,c,h',w'}
\right)+b_o.
~~~

h' 和 w' 由 stride、padding 和 dilation 决定，`b_o` 是输出通道 `o` 的 bias，在所有输入通道和卷积窗口求和完成后只加一次。实际框架还会包含边界处理和通道分组，但核心直觉是：同一组局部权重在不同空间位置重复使用。

卷积的权重共享带来两个重要性质：

1. 一个边缘检测模式可以在图片不同位置复用。
2. 局部邻域在网络早期被优先建模。

这就是 CNN 的局部归纳偏置。它不是“模型知道所有物体”，而是结构先假设局部邻域有价值。

### 2.2 输出尺寸与 stride

对单个空间维度，卷积输出大小可写为：

~~~math
H_{\mathrm{out}}
=\left\lfloor
\frac{H+2p-d(K-1)-1}{s}+1
\right\rfloor,
~~~

其中 H 是输入尺寸，p 是 padding，K 是 kernel size，s 是 stride，d 是 dilation。宽度方向使用同样公式。

这个公式说明，视觉特征图的空间尺寸不是由模型名称决定的，而是由卷积配置和输入尺寸共同决定的。stride 越大，空间分辨率下降越快；连续下采样可以降低后续计算，却会损失小目标定位精度。

### 2.3 感受野

如果一个输出单元依赖输入中的一块区域，这块区域称为它的感受野。堆叠卷积后，感受野逐层扩大。对一维简化情况，若第 l 层 kernel 为 k_l、stride 为 s_l，跳距 j_l 和感受野 r_l 可以递推：

~~~math
j_l=j_{l-1}s_l,\qquad
r_l=r_{l-1}+(k_l-1)j_{l-1},
~~~

初始 j_0=1、r_0=1。这个递推有两个工程含义：

1. 早期层更适合局部纹理和边缘。
2. 深层单元覆盖更大的输入区域，语义更强但空间精度更粗。

不能只看最后一层的通道数判断它是否适合 OCR；还要看最后一层的空间步幅和小文字在输入中占多少像素。

### 2.4 残差连接

ResNet 的核心形式是：

~~~math
h_{l+1}=h_l+F_l(h_l).
~~~

残差块让网络学习相对输入的增量，而不是每一层都从零学习完整变换。反向传播时，恒等路径为梯度提供了直接通道，有助于训练更深的网络。

残差连接并不会自动增加分辨率，也不会自动保留 OCR 细节。它解决的是优化和表示变换的组织方式，空间下采样仍由 stride、pooling 和具体 block 决定。

### 2.5 CNN 的多尺度输出

检测、分割和文档布局任务常从 CNN 的不同 stage 取特征：

~~~text
stage_1: 高分辨率、低层纹理
stage_2: 较高分辨率、局部形状
stage_3: 中等分辨率、部件语义
stage_4: 低分辨率、全局语义
~~~

这类 feature pyramid 的价值是同时保留不同尺度的信息。一个只输出最终 global pooling 向量的 CNN，可能很适合分类，却不适合直接预测细粒度坐标。

### 2.6 CNN 与 ViT 的比较边界

不能把“CNN 局部、ViT 全局”说成绝对结论。深层 CNN 也可以拥有很大的感受野，ViT 也依赖 patch 划分和位置表示。更准确的比较是：

1. CNN 在结构中直接编码局部性和权重共享。
2. ViT 把局部块离散成 token，再让 attention 建立跨块关系。
3. CNN 的空间计算通常随 feature map 面积线性增长。
4. 全局 ViT attention 的 token 交互通常带有平方项。
5. 层级 ViT、窗口 attention 和混合架构会改变这些成本关系。

模型选择要绑定输入尺寸、目标任务和硬件，而不是只比较论文中的参数量。

## 3. ViT：把图像变成 patch 序列

### 3.1 Patch 网格的三种口径

设输入高度 H、宽度 W、patch size P。工程中常见三种口径：

1. 先 resize 或 crop 到 H、W 都能被 P 整除。
2. 显式 padding 到完整网格。
3. 不 padding，直接使用 stride=P 的卷积，舍弃边缘不足一个 patch 的区域。

前两种完整网格的 token 数为：

~~~math
N_h=\left\lceil\frac{H}{P}\right\rceil,\qquad
N_w=\left\lceil\frac{W}{P}\right\rceil,\qquad
N=N_hN_w.
~~~

第三种无 padding 的卷积输出为：

~~~math
N_h=\left\lfloor\frac{H-P}{P}\right\rfloor+1,\qquad
N_w=\left\lfloor\frac{W-P}{P}\right\rfloor+1.
~~~

例如 H=W=225、P=14：

1. 显式 padding 或先补齐时是 17 x 17=289 个 patch。
2. 无 padding 的 kernel=14、stride=14 卷积是 16 x 16=256 个 patch。

两者都可以是合理设计，但不能在同一份文档里把它们的 token 数混用。模型卡、processor 和实际代码决定了最终口径。

### 3.2 Patch 是什么

一个 RGB patch 的原始形状是：

~~~math
p_i\in\mathbb{R}^{C\times P\times P}.
~~~

展平后：

~~~math
\operatorname{vec}(p_i)\in\mathbb{R}^{CP^2}.
~~~

对每个 patch 使用线性投影：

~~~math
z_i=\operatorname{vec}(p_i)W_e+b_e,
\qquad
W_e\in\mathbb{R}^{CP^2\times d_v}.
~~~

z_i 是第 i 个视觉 token，d_v 是视觉主干的 hidden size。这个操作并没有理解图片内容，它只是把局部像素块放入一个可学习的向量空间；语义关系由后续主干和预训练目标建立。

### 3.3 Conv2d 是批量 patch 投影

线性 patch embedding 可以用卷积高效实现：

~~~text
Conv2d(
    in_channels=C,
    out_channels=d_v,
    kernel_size=P,
    stride=P
)
~~~

卷积权重形状是：

~~~math
W_{\mathrm{conv}}\in\mathbb{R}^{d_v\times C\times P\times P}.
~~~

它与线性层的参数数量相同：

~~~math
\#\mathrm{params}
=d_vCP^2+d_v
~~~

其中最后一项是 bias。卷积实现的优势是直接在整张图片上并行计算，不需要先在 Python 中逐块切片。

### 3.4 Token 顺序就是空间契约

假设 patch 网格为 N_h x N_w，常见的展平顺序是按行优先：

~~~math
\operatorname{row}(i)=\left\lfloor\frac{i}{N_w}\right\rfloor,\qquad
\operatorname{col}(i)=i\bmod N_w.
~~~

如果第 0 个 token 是 CLS，那么 patch token 的序号应使用 i-1 后再映射坐标。下游模块如果假设列优先、加入了特殊 token 或删除了某些 patch，坐标就会错位。

对一个 patch token，中心坐标可以近似写成：

~~~math
x_i=\left(\operatorname{col}(i)+\frac{1}{2}\right)P,\qquad
y_i=\left(\operatorname{row}(i)+\frac{1}{2}\right)P.
~~~

这只是规则网格的几何中心，不是模型真正关注的像素区域。它仍然是建立 grounding、裁剪回原图和调试 token 顺序的必要索引。

如果输入为了形成完整网格而做了 padding，这个中心坐标首先属于“补齐后的输入画布”。要映射回原图，还需要保存 padding 的左、上偏移以及 resize 比例；不能把 `(x_i,y_i)` 直接当作原始文件坐标。切块输入还要再加上 tile 在原图中的 offset。

### 3.5 加入 CLS 与位置表示

如果使用 CLS token，初始序列可以写成：

~~~math
X_0=
[x_{\mathrm{cls}};z_1;\ldots;z_N]
+E_{\mathrm{pos}},
~~~

其中 E_pos 的形状必须与序列长度和 hidden size 匹配：

~~~math
E_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times d_v}.
~~~

位置表示的责任是让模型区分“同一个视觉 token 出现在左上角”和“出现在右下角”。常见实现包括：

1. 学习的绝对位置 embedding。
2. 二维分解或二维插值的位置 embedding。
3. 相对位置偏置。
4. 旋转或其他 attention 内位置表示。

不同位置方案的外推行为不同。将 224 的位置表直接用于 336 的网格，通常需要插值或模型专门支持动态分辨率；不能把位置表当作一个与分辨率无关的常量。

### 3.6 ViT 主干的形状

忽略 batch 之外的实现细节，一个 Transformer encoder block 保持：

~~~math
X_l\in\mathbb{R}^{B\times L\times d_v}
\quad\longrightarrow\quad
X_{l+1}\in\mathbb{R}^{B\times L\times d_v},
~~~

其中 L=N 或 N+1。self-attention 在 token 之间交换信息，MLP 在每个 token 内进行通道变换。除非使用层级结构或 token merging，序列长度通常保持不变。

## 4. 全局向量、Patch Tokens 与多层特征

### 4.1 CLS token

CLS token 是一个可学习的特殊 token。它通过 attention 与其他 patch 交互，最终可以作为整张图片的摘要：

~~~math
g=x_{\mathrm{cls}}^{(L)}.
~~~

它的优点是输出固定长度，适合分类和图文对齐；它的风险是细粒度空间信息被压缩进一个向量，后续很难可靠恢复某个小区域的文字。

### 4.2 平均池化与 attention pooling

如果没有 CLS，或实现选择对 patch tokens 池化，可以写成：

~~~math
g=\frac{1}{N}\sum_{i=1}^{N}z_i^{(L)}.
~~~

attention pooling 则用一组权重聚合：

~~~math
g=\sum_{i=1}^{N}a_i z_i^{(L)},\qquad
\sum_i a_i=1.
~~~

不同 pooling 会改变图文 embedding 的统计分布。加载一个公开视觉塔时，不能只取最后一个 token 就假设等价于原模型的 image embedding。

### 4.3 Patch tokens

VLM 更常需要：

~~~math
Z=\left[z_1^{(L)},\ldots,z_N^{(L)}\right]
\in\mathbb{R}^{B\times N\times d_v}.
~~~

Patch tokens 保留网格顺序，可以被 projector 转换后插入语言模型的多模态序列：

~~~math
Y=ZW_p+b_p,\qquad
Y\in\mathbb{R}^{B\times N\times d_l}.
~~~

projector 改变通道维，不会凭空增加已经在视觉塔中丢失的像素细节。若 N 个 token 在输入阶段已经合并成 64 个，projector 不会把它们还原成 576 个独立区域。

### 4.4 多层特征

视觉主干的不同层通常有不同的表示倾向：

| 层级 | 常见信息 | 适合问题 |
| --- | --- | --- |
| 浅层 | 边缘、纹理、颜色和局部模式 | 低级视觉与细节 |
| 中层 | 部件、形状和局部组合 | 图表、区域和细粒度属性 |
| 深层 | 全局对象和语义关系 | 检索、分类和描述 |

这不是严格的定律。训练目标、架构、输入分辨率和 pooling 都会改变层间差异。选择某一层时应使用任务验证集和反事实样本，而不是只因为“最后一层语义最强”就丢掉中间层。

### 4.5 全局与局部的系统分工

一个高效的系统可以把不同表示放到不同阶段：

~~~text
global embedding -> 便宜召回与粗分类
patch tokens    -> 少量候选的细粒度问答
OCR / regions   -> 字段、坐标和引用验证
~~~

这样做的前提是召回阶段不能漏掉真正相关的候选。如果全局 embedding 对小字完全不敏感，后续再强的 OCR 也没有候选页面可处理。因此粗召回和精确验证应分别评估。

## 5. 位置、分辨率与 token 预算

### 5.1 Token 数按面积增长

在完整网格且 patch size 固定时：

~~~math
N=
\left\lceil\frac{H}{P}\right\rceil
\left\lceil\frac{W}{P}\right\rceil.
~~~

如果高度和宽度同时乘以 r，token 数大约乘以 r^2。以 patch=14 的补齐网格为例：

~~~text
224 x 224 -> 16 x 16 = 256
336 x 336 -> 24 x 24 = 576
448 x 448 -> 32 x 32 = 1024
~~~

若模型使用无 padding 并且输入尺寸不能整除 P，上表要改成 floor 公式。工程报告必须注明是补齐网格还是卷积有效区域。

### 5.2 Attention 与 MLP 成本

对长度 L、hidden size d_v 的全局 self-attention，注意力分数矩阵的元素数量约为：

~~~math
M_{\mathrm{score}}=L^2.
~~~

如果还考虑每个 head 的维度，QK^T 和 attention value 的主要乘法量可近似写成：

~~~math
C_{\mathrm{attn}}\propto L^2d_v.
~~~

Transformer MLP 的计算量更接近：

~~~math
C_{\mathrm{mlp}}\propto Ld_vd_{\mathrm{ff}}.
~~~

因此在高分辨率视觉塔中，attention 的平方项会成为重要成本来源，但实际瓶颈仍取决于 head 数、MLP 宽度、kernel、fused backend、显存带宽和 batch size。公式用于量级判断，不是硬件吞吐承诺。

### 5.3 为什么分辨率放大很贵

若 H、W 和 token 数都近似按 r 放大：

~~~math
N'\approx r^2N,\qquad
M'_{\mathrm{score}}\approx r^4M_{\mathrm{score}}.
~~~

例如 224 到 448，边长乘以 2，patch token 约乘以 4，attention score cell 约乘以 16。若把视觉 token 继续送入 LLM，语言模型的 prefill 也会增加；视觉塔和语言主干的成本不能只看其中一层。

### 5.4 视觉 token 进入多模态上下文

若文本 token 数为 T_text、视觉 token 数为 N_vis、额外特殊 token 数为 T_extra，则：

~~~math
T_{\mathrm{context}}
=T_{\mathrm{text}}+N_{\mathrm{vis}}+T_{\mathrm{extra}}.
~~~

如果有多张图片或多个裁剪：

~~~math
N_{\mathrm{vis}}
=\sum_{m=1}^{M}N_m.
~~~

视觉 token 预算必须和回答长度、KV cache、batch 并发一起估算。一个 1024 token 的单图看似不大，但在 32 张图片、长文本和高并发场景中会迅速成为上下文瓶颈。

### 5.5 高分辨率的工程策略

常见策略及其信息损失不同：

1. 动态 resize：按图片内容选择输入尺寸，节省平均成本，但请求间延迟不稳定。
2. 切块或 tiling：保留局部细节，却需要合并跨块坐标和重复边缘。
3. 多尺度编码：同时看缩略图和局部裁剪，成本较高但更完整。
4. token merging：合并相似 token，便宜但可能合并掉小目标。
5. token pruning：保留模型认为重要的 token，风险是重要性判断本身可能漏掉证据。
6. 先 OCR/检测再局部编码：适合文档，流水线更复杂。

任何压缩都应绑定评估切片。对图像描述，合并背景 token 可能没有影响；对金额、车牌和表格，恰恰是低频小区域最重要。

### 5.6 位置编码的分辨率迁移

如果位置 embedding 是固定长度 L_0，而目标网格长度是 L_1，就不能直接相加。常见处理包括：

1. 将二维位置表恢复成网格后做二维插值。
2. 使用相对位置或可外推的位置机制。
3. 在预训练阶段覆盖多种分辨率。
4. 重新训练适配新网格的参数。

插值只改变位置参数，不会补回原始训练中没有见过的细粒度视觉模式。位置迁移通过不代表新分辨率上的 OCR 能力也通过。

## 6. 预训练目标与视觉塔的能力

### 6.1 监督分类预训练

传统视觉塔可以使用类别标签训练：

~~~math
L_{\mathrm{cls}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log p(y_i\mid x_i).
~~~

它会鼓励表示区分标注类别。优点是任务目标明确，缺点是类别空间受限，且标签粒度由数据集决定。

### 6.2 图文对齐预训练

CLIP 风格目标让图像和文本进入共享空间。视觉塔因此学到与自然语言概念相关的全局特征，适合检索和 zero-shot 分类。它不自动提供像素到词语的精确对应，也不保证小字号文字能被解码。

### 6.3 Sigmoid 配对目标

SigLIP 路线将每个图文对作为匹配或不匹配的二分类关系，避免把整个 batch 的候选归一化成一个 softmax 分类问题。其损失形式可以抽象为：

~~~math
L_{\mathrm{sigmoid}}
=-\frac{1}{M}
\sum_{(i,j)}
\log\sigma\left(y_{ij}\,s_{ij}\right),
\qquad y_{ij}\in\{-1,+1\}.
~~~

这里的 M 是配对数量，s_ij 是缩放后的相似度。不同实现会对正负配对采样、权重和标签组织做具体设计。本章只借此说明：视觉塔的训练目标会影响它如何组织共享空间，不能因为都叫 image encoder 就假设特征可直接互换。

### 6.4 重建和掩码预训练

MAE 等方法随机遮挡图像 patch，让编码器或解码器重建被遮挡内容。它更直接地利用视觉结构学习表示，但训练目标与图文对齐不同。一个 MAE 预训练的视觉塔可能在视觉迁移上很强，却未必已经位于适合文字相似度的共享空间。

因此下游接入 CLIP 或 VLM 时要确认：

1. 视觉塔是否已经和语言对齐。
2. 是否需要额外 projection 或对齐训练。
3. 取哪个层、哪种 pooling。
4. 预处理和输入分辨率是否匹配。

### 6.5 冻结与微调

VLM 训练常见三种阶段：

1. 冻结视觉塔和语言模型，只训练 projector。
2. 冻结语言模型，微调视觉塔后几层和 projector。
3. 使用低学习率联合微调视觉塔、projector 和语言模型。

冻结可以保留原有视觉能力并降低显存，代价是适应领域图像和 OCR 的能力有限。联合微调适应性更强，但可能破坏原有图文空间、增加灾难性遗忘和过拟合风险。验证时要同时看目标任务和原始对齐能力。

## 7. 预处理是视觉编码器的输入契约

### 7.1 Resize、crop 与 padding

同一原图可以被处理成不同输入：

~~~text
原图 1200 x 800
    -> shortest-edge resize + center crop
    -> 224 x 224

原图 1200 x 800
    -> 等比例缩放 + padding
    -> 336 x 336

原图 1200 x 800
    -> 切成多个局部窗口
    -> 若干个 336 x 336
~~~

中心裁剪可能切掉边缘文字，padding 可能改变背景分布，切块会产生多个局部坐标。输入尺寸相同不意味着预处理相同。

### 7.2 数值范围和归一化

像素可能以 [0,255] 的整数、[0,1] 的浮点或经过均值方差标准化的浮点进入模型。若每个通道的均值为 mu_c、标准差为 sigma_c，则：

~~~math
x'_{c,h,w}
=\frac{x_{c,h,w}-\mu_c}{\sigma_c}.
~~~

均值和标准差必须与预训练实现一致。颜色通道从 RGB 换成 BGR，或把已经归一化的输入再次归一化，都会改变视觉特征。

### 7.3 非正方形图片

把任意图片强行拉伸到正方形会改变物体比例；保持比例再 padding 会引入空白区域；切块可以保留细节但增加 token。应根据任务选择：

1. 分类和粗检索可以接受固定 crop。
2. 文档和图表通常需要保留版面比例。
3. OCR 需要保存原图到输入坐标的映射。
4. 定位任务要记录 crop offset 和缩放因子。

如果模型输出一个区域坐标，坐标必须明确是原图、resize 图还是某个 tile 的局部坐标。

### 7.4 Processor 的版本化

使用公开权重时，processor 往往包含 resize、crop、normalize、颜色和特殊尺寸策略。工程上应保存：

1. processor 名称和版本。
2. 输入尺寸与插值方式。
3. crop 或 padding 模式。
4. mean、std、颜色顺序。
5. 多图、多 crop 和拼接规则。

不要在生产环境中只保存一个模型权重文件而丢掉这些配置。

## 8. OCR、图表和空间细节为什么困难

### 8.1 小文字的像素预算

若一个字符的笔画在原图中只有若干像素，resize 到模型输入后可能小于一个 patch 的局部有效区域。patch embedding 会将同一 patch 中的文字、背景和边框混合成一个向量，后续主干不一定能恢复每个字符。

这不是单纯的语言模型问题。文字在视觉输入阶段已经被缩放、裁剪或压缩，LLM 只能基于剩余表示生成结果。

### 8.2 图表需要多种关系

读图表至少需要：

1. 文字标签。
2. 坐标轴和刻度。
3. 图例与颜色绑定。
4. 数据点和趋势。
5. 页面中的相对位置。

全局向量可以判断“这是一张销售折线图”，却不一定能恢复某个季度的精确数值。图表问答通常需要更高分辨率、OCR、区域特征或结构化解析。

### 8.3 多个对象的关系

“红球在蓝盒子上方”和“蓝盒子在红球上方”包含相同对象词，却是相反关系。全局 pooling 和低分辨率 patch 可能保留对象存在，却丢掉主客体、相对位置或遮挡关系。

反事实测试可以：

1. 交换两个对象的文字描述。
2. 改变图片中对象的相对位置。
3. 保持背景和颜色不变，只改变关系。
4. 分别测试全局向量和 patch token 的变化。

如果表示对关键关系几乎不变，不能用语言输出的流畅度掩盖视觉证据不足。

### 8.4 可靠的补救策略

对于细粒度任务，可以组合：

~~~text
缩略图全局编码
    -> 召回相关图片或页面
局部裁剪 / OCR / layout
    -> 提供精确字段和坐标
VLM 或规则验证
    -> 生成带引用的结果
~~~

增加分辨率只是一个选项，不是所有问题的答案。外部 OCR 可能带来识别错误，切块可能丢跨块关系，额外模型也会增加延迟；每个补救策略都需要自己的验证集。

## 9. 视觉特征选择与架构取舍

### 9.1 选择全局向量还是 patch tokens

可用下面的判断：

| 任务 | 首选表示 | 主要原因 |
| --- | --- | --- |
| 图片库主题检索 | 全局 embedding | 固定长度、索引便宜 |
| zero-shot 粗分类 | 全局 embedding | 文本候选可直接比较 |
| 物体位置和区域引用 | patch tokens 或多尺度特征 | 保留空间网格 |
| 扫描合同字段 | OCR、局部视觉特征和 patch tokens | 需要文字与版面 |
| 多轮视觉对话 | patch tokens 加压缩策略 | 需要给语言模型提供细节 |

这张表不是绝对规则。真正系统往往采用全局召回加局部重排的组合。

### 9.2 CNN、全局 ViT 与层级 ViT

| 路线 | 典型优势 | 主要成本或风险 |
| --- | --- | --- |
| CNN / ResNet | 局部先验、多尺度和成熟部署 | 与语言 Transformer 接口不同 |
| 全局 ViT | 全局交互、结构统一、适合大规模预训练 | token 平方成本和数据敏感性 |
| 窗口或层级 ViT | 更高分辨率下控制 attention 成本 | 跨窗口关系和实现更复杂 |
| 混合架构 | 综合卷积与 attention 的优点 | 组件、权重和预处理契约更多 |

Swin 等层级方法通过窗口或分层改变全局 attention 的成本曲线；这不意味着所有高分辨率输入都变便宜，而是把交互范围、下采样和跨窗口通信重新分配。

### 9.3 选择视觉塔的实验矩阵

对候选 vision encoder，至少比较：

1. 全局检索 Recall@K。
2. 细粒度属性和关系准确率。
3. OCR 字符或字段准确率。
4. 输入分辨率与 token 数。
5. vision encoder 延迟和峰值显存。
6. projector 后的 VLM 任务质量。
7. 语言、领域和高风险切片。

参数量小不等于端到端更快。一个视觉塔输出四倍 token，可能让语言模型 prefill 成为主要瓶颈。

### 9.4 视觉特征的版本迁移

替换视觉塔时，不能只保持输出维度相同。还要核对：

1. token 数和特殊 token。
2. token 的空间顺序。
3. feature 的归一化与均值方差。
4. projector 训练是否依赖旧特征分布。
5. 位置编码和动态分辨率。
6. 训练数据与下游评估是否重新校准。

如果新旧视觉塔都输出 [B, 576, 1024]，也不代表它们的 token 语义和尺度相同。

## 10. Vision Encoder 与 VLM 的接口

### 10.1 Projector 的最小作用

设视觉 token 为：

~~~math
Z\in\mathbb{R}^{B\times N\times d_v},
~~~

语言模型 hidden size 为 d_l。线性 projector：

~~~math
Y=ZW_p+b_p,\qquad
W_p\in\mathbb{R}^{d_v\times d_l},
\qquad
Y\in\mathbb{R}^{B\times N\times d_l}.
~~~

它解决的是通道维度对齐。一个 MLP projector 可以增加非线性，但仍然不能替代视觉塔的空间编码，也不能把已经删除的 token 恢复出来。

### 10.2 视觉 token 如何进入语言序列

多模态语言模型可能把视觉 token 插入文本 token 的某个位置：

~~~text
system / user text
    -> image placeholder
    -> projected visual tokens
    -> remaining text
    -> language model
~~~

实际实现还要定义：

1. placeholder 数量与视觉 token 数的对应。
2. 多图之间的边界标记。
3. 视觉 token 的位置编码或模态标识。
4. 视觉 token 是否参与 causal mask 的某些路径。
5. loss 是否只计算 assistant 文本。

视觉塔输出对了，不代表 chat template 和 label mask 就对了。

### 10.3 冻结阶段的诊断

如果只训练 projector，出现以下现象时，根因可能在不同层：

| 现象 | 可能原因 |
| --- | --- |
| loss 不下降 | placeholder、dtype、shape 或 label mask 错 |
| 全局描述可用，OCR 失败 | 分辨率、patch 信息或视觉塔目标不适合 |
| 视觉问答像纯文本猜测 | visual tokens 未真正进入主干或训练监督弱 |
| 训练集好、跨域差 | 视觉塔和目标图像分布不匹配 |
| 多图顺序混乱 | token 边界、位置或样本拼接错 |

调试时应保存一批固定图片、视觉 token 统计、projector 输出范数和遮挡前后结果。

### 10.4 视觉信息与语言先验的分离

让语言模型回答图片问题时，至少做三种对照：

1. 原图和正确问题。
2. 关键区域遮挡但问题不变。
3. 替换成语义相反的图片但问题不变。

如果三种输入得到几乎相同的回答，说明输出可能主要来自语言先验。这个测试不能证明模型完全不使用图像，但比只看回答流畅度更接近证据检查。

## 11. 贯穿案例：扫描合同的多级视觉编码

### 11.1 业务约束

企业知识库中的合同页面具有以下特点：

1. 页面尺寸和纵横比不统一。
2. 关键条款可能只有一行小字。
3. 表格、印章、页眉和脚注共同出现。
4. 用户问题同时包含主题检索和数字条件。
5. 返回结果必须带页码、区域和权限。

一个只把每页压缩成单个全局 embedding 的系统，可能能找出“合同”主题，却不能保证找对期限、金额和表格行。

### 11.2 两级或三级表示

可以采用：

~~~text
页面缩略图
    -> 全局 vision encoder -> 主题候选召回
候选页面
    -> 高分辨率 tile / patch tokens -> 局部视觉理解
局部区域
    -> OCR + layout + 规则核验 -> 字段和引用
~~~

全局阶段追求召回，局部阶段追求细节，OCR 和规则阶段追求可验证字段。每一级都应保存输入坐标到原图的映射。

### 11.3 失败案例

如果页面缩略图中的“60 天”被缩成不可见的小字，CLIP 或全局视觉塔可能只判断页面与付款有关；局部候选没有被召回，后面的 OCR 根本没有机会运行。

另一个失败是 tile 识别出“60 天”，但没有保留 tile 在原页面中的 offset，最后引用了错误区域。视觉编码准确不等于证据链完整。

### 11.4 评估分母

应该分别报告：

1. 页面级候选召回率。
2. 局部区域召回率。
3. OCR 字段准确率。
4. 坐标映射准确率。
5. 引用支持率。
6. 每个成功查询的视觉 token、GPU 时间和存储成本。

一个总体 VLM 分数不能替代这些层级指标，因为它无法说明错误是在缩略图、tile、OCR、坐标还是权限过滤中产生的。

## 12. 最小可运行 Vision Encoder 成本与形状审计

下面的 demo 不依赖第三方库。它把两种网格口径明确分开，并计算：

1. padding 网格与无 padding 卷积网格。
2. patch token、CLS token 和位置表形状。
3. patch embedding 与 projector 参数量。
4. token 数和 attention score cell 的增长。
5. token 到二维坐标的映射。
6. 不同任务对分辨率的需求提示。

程序只验证算术和接口，不验证任何真实视觉模型的识别能力。

~~~python
from math import ceil


def padded_grid(height, width, patch):
    if min(height, width, patch) <= 0:
        raise ValueError("image dimensions and patch must be positive")
    return ceil(height / patch), ceil(width / patch)


def valid_conv_grid(height, width, patch):
    if min(height, width, patch) <= 0:
        raise ValueError("image dimensions and patch must be positive")
    if height < patch or width < patch:
        raise ValueError("an unpadded patch grid needs at least one full patch")
    return (height - patch) // patch + 1, (width - patch) // patch + 1


def patch_tokens(height, width, patch, padded=True):
    grid_h, grid_w = (
        padded_grid(height, width, patch)
        if padded
        else valid_conv_grid(height, width, patch)
    )
    return grid_h * grid_w


def patch_vector_dim(channels, patch):
    if min(channels, patch) <= 0:
        raise ValueError("channels and patch must be positive")
    return channels * patch * patch


def patch_embedding_params(channels, patch, hidden, bias=True):
    if hidden <= 0:
        raise ValueError("hidden size must be positive")
    params = patch_vector_dim(channels, patch) * hidden
    return params + (hidden if bias else 0)


def position_shape(height, width, patch, hidden, cls=True, padded=True):
    if hidden <= 0:
        raise ValueError("hidden size must be positive")
    tokens = patch_tokens(height, width, patch, padded=padded)
    return (tokens + (1 if cls else 0), hidden)


def attention_cells(tokens):
    if tokens <= 0:
        raise ValueError("token count must be positive")
    return tokens * tokens


def projector_params(vision_hidden, llm_hidden, bias=True):
    if vision_hidden <= 0 or llm_hidden <= 0:
        raise ValueError("projector dimensions must be positive")
    params = vision_hidden * llm_hidden
    return params + (llm_hidden if bias else 0)


def token_coordinate(index, grid_width, patch, cls=True, grid_height=None):
    if index < 0 or grid_width <= 0 or patch <= 0:
        raise ValueError("token index, grid width and patch must be valid")
    patch_index = index - 1 if cls else index
    if patch_index < 0:
        raise ValueError("CLS token has no patch coordinate")
    if grid_height is not None and grid_height <= 0:
        raise ValueError("grid height must be positive")
    if grid_height is not None and patch_index >= grid_height * grid_width:
        raise ValueError("patch index is outside the grid")
    row, column = divmod(patch_index, grid_width)
    return {
        "row": row,
        "column": column,
        "center_y": (row + 0.5) * patch,
        "center_x": (column + 0.5) * patch,
    }


resolutions = [
    {"name": "224_square", "height": 224, "width": 224},
    {"name": "336_square", "height": 336, "width": 336},
    {"name": "448_square", "height": 448, "width": 448},
    {"name": "672_by_448", "height": 672, "width": 448},
]
patch = 14
channels = 3
vision_hidden = 1024
llm_hidden = 4096

resolution_table = []
for item in resolutions:
    grid = padded_grid(item["height"], item["width"], patch)
    tokens = patch_tokens(item["height"], item["width"], patch, padded=True)
    sequence_length = tokens + 1
    resolution_table.append(
        {
            "name": item["name"],
            "grid": grid,
            "patch_tokens": tokens,
            "with_cls": sequence_length,
            "attn_cells": attention_cells(sequence_length),
        }
    )

base_cells = resolution_table[0]["attn_cells"]
for row in resolution_table:
    row["attn_vs_224"] = round(row["attn_cells"] / base_cells, 2)

shape_audit = {
    "padded_225_patch14": padded_grid(225, 225, 14),
    "valid_225_patch14": valid_conv_grid(225, 225, 14),
    "padded_336_patch14": padded_grid(336, 336, 14),
    "patch_vector_dim": patch_vector_dim(channels, patch),
    "patch_weight": (vision_hidden, channels, patch, patch),
    "position_embedding": position_shape(
        336,
        336,
        patch,
        vision_hidden,
        cls=True,
        padded=True,
    ),
    "projected_tokens": (
        2,
        patch_tokens(336, 336, patch, padded=True),
        llm_hidden,
    ),
    "token_25_coordinate": token_coordinate(
        index=25,
        grid_width=24,
        grid_height=24,
        patch=patch,
        cls=True,
    ),
}

parameter_audit = {
    "patch_embedding_with_bias": patch_embedding_params(
        channels,
        patch,
        vision_hidden,
        bias=True,
    ),
    "projector_with_bias": projector_params(
        vision_hidden,
        llm_hidden,
        bias=True,
    ),
}

task_tradeoffs = [
    {
        "name": "image_caption",
        "detail_need": 0.35,
        "tokens": patch_tokens(224, 224, patch, padded=True),
    },
    {
        "name": "chart_qa",
        "detail_need": 0.75,
        "tokens": patch_tokens(336, 336, patch, padded=True),
    },
    {
        "name": "dense_ocr",
        "detail_need": 0.92,
        "tokens": patch_tokens(448, 448, patch, padded=True),
    },
]
for item in task_tradeoffs:
    item["needs_high_resolution"] = item["detail_need"] >= 0.7
    item["over_1k_tokens"] = item["tokens"] > 1000

checks = {
    "padding_and_valid_are_distinct": (
        shape_audit["padded_225_patch14"] == (17, 17)
        and shape_audit["valid_225_patch14"] == (16, 16)
    ),
    "padded_336_grid_ok": (
        shape_audit["padded_336_patch14"] == (24, 24)
    ),
    "position_shape_ok": (
        shape_audit["position_embedding"] == (577, 1024)
    ),
    "projector_shape_ok": (
        shape_audit["projected_tokens"] == (2, 576, 4096)
    ),
    "attention_cost_grows": (
        resolution_table[2]["attn_vs_224"]
        > resolution_table[1]["attn_vs_224"]
        > 1.0
    ),
    "coordinate_mapping_ok": (
        shape_audit["token_25_coordinate"]["row"] == 1
        and shape_audit["token_25_coordinate"]["column"] == 0
    ),
    "ocr_cost_is_visible": (
        task_tradeoffs[2]["needs_high_resolution"]
        and task_tradeoffs[2]["over_1k_tokens"]
    ),
}

signals = []
actions = []
if not checks["padding_and_valid_are_distinct"]:
    signals.append("padding and valid-convolution grids disagree with the contract")
    actions.append("inspect resize, padding and patch convolution settings")
if not checks["position_shape_ok"]:
    signals.append("position embedding length does not match the visual sequence")
    actions.append("inspect grid dimensions, CLS token and interpolation")
if not checks["coordinate_mapping_ok"]:
    signals.append("patch index does not map to the expected grid coordinate")
    actions.append("inspect token order and special-token offset")
if not checks["ocr_cost_is_visible"]:
    signals.append("the high-resolution OCR scenario was not identified")
    actions.append("inspect task requirements and token budget")

decision = (
    "continue_to_encoder_benchmark"
    if all(checks.values())
    else "repair_visual_shape_contract"
)

print("resolution_table=", resolution_table)
print("shape_audit=", shape_audit)
print("parameter_audit=", parameter_audit)
print("task_tradeoffs=", task_tradeoffs)
print("signals=", signals)
print("actions=", actions)
print("checks=", checks)
print("decision=", decision)
~~~

按代码中的补齐网格口径，224、336、448 的 patch token 分别是 256、576、1024；336 的 CLS 加 patch 序列长度是 577，projector 输出形状是 (2, 576, 4096)。对 225 x 225 图片使用 patch=14 时，补齐网格有 289 个 token，无 padding 卷积有 256 个 token，程序把两种口径同时打印出来，避免把 ceil 和 floor 混为一谈。

这个实验没有运行 Transformer，也没有读取真实图片。它只证明形状、参数量、位置索引和成本估算彼此一致。真实视觉编码器还需要在目标分辨率、目标硬件和任务切片上评估。

## 13. 练习：把视觉表示和任务联系起来

1. 对 225 x 225、patch=14 的图片，分别计算无 padding 和补齐网格的 token 数。
2. 推导 Conv2d 的输出尺寸公式，并说明 stride、padding 和 dilation 的作用。
3. 设 patch 网格为 24 x 24 且有 CLS token，计算第 25 个序号对应的行列坐标。
4. 比较 CLS、平均池化和 patch tokens 在图像检索、OCR、定位中的责任。
5. 说明 224 到 448 的 token 数和全局 attention score cell 如何变化。
6. 为扫描合同设计缩略图、tile 和 OCR 三阶段的坐标映射。
7. 选择一个 CNN 和一个 ViT，列出它们的输入预处理、输出层、空间步幅和参数版本。
8. 解释为什么位置 embedding 从 224 插值到 336 后，不能直接宣称 OCR 能力提升。
9. 设计遮挡、换背景和交换对象位置三个反事实实验，比较全局向量和 patch token 的变化。
10. 计算视觉 token 为 576、文本 token 为 2048、回答预算为 1024 时的上下文占用，并讨论多图场景。
11. 设计冻结视觉塔、只训练 projector 的最小实验，列出应记录的 shape、范数和 loss。
12. 为 CLIP 召回加局部 OCR 重排的合同检索系统写出页面召回率、字段准确率和引用支持率的分母。

## 14. 资料入口与证据边界

1. He et al., Deep Residual Learning for Image Recognition，ResNet 原始论文：<https://arxiv.org/abs/1512.03385>。
2. Dosovitskiy et al., An Image is Worth 16x16 Words，ViT 原始论文：<https://arxiv.org/abs/2010.11929>。
3. Radford et al., Learning Transferable Visual Models From Natural Language，CLIP 原始论文：<https://arxiv.org/abs/2103.00020>。
4. Zhai et al., Sigmoid Loss for Language Image Pre-Training，SigLIP 原始论文：<https://arxiv.org/abs/2303.15343>。
5. He et al., Masked Autoencoders Are Scalable Vision Learners，MAE 原始论文：<https://arxiv.org/abs/2111.06377>。
6. Liu et al., Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows：<https://arxiv.org/abs/2103.14030>。
7. OpenAI CLIP 官方代码：<https://github.com/openai/CLIP>。

这些资料分别支持残差 CNN、patch Transformer、图文对齐、sigmoid 配对目标、掩码视觉预训练、层级窗口 attention 和某一公开 CLIP 实现。它们不能直接证明任意视觉塔的 OCR、文档理解、VLM 对话质量、吞吐、价格或安全率。模型卡和 processor 文档中的分辨率、pooling、特殊 token 与实际权重版本必须绑定记录。

当一个视觉塔被描述为“高分辨率”“支持动态尺寸”或“适合 OCR”时，应继续核对：

1. 高分辨率是 resize、padding、tiling 还是多尺度输入。
2. token 数和位置编码如何变化。
3. 返回的是全局向量、patch tokens 还是多层特征。
4. OCR 评估是字符、字段、区域还是端到端回答。
5. 小字、表格、旋转文本和低资源语言是否有单独切片。
6. 结果是否使用了外部 OCR 或规则后处理。
7. 成本是否包含视觉塔、LLM prefill、缓存和索引。

## 15. 本章回顾

Vision encoder 把像素转换为全局向量、patch tokens 或多层视觉特征。CNN 用局部卷积、权重共享和层级下采样建立视觉归纳偏置；ViT 用 patch embedding 把图像变成序列，再用位置表示和 Transformer 建模跨区域关系。

输入尺寸、patch size、padding 和 token 顺序共同决定视觉 shape。补齐网格使用 ceil，无 padding 的 stride patch 卷积使用 floor；两种口径必须分开。CLS 或 pooling 适合固定长度的全局任务，patch tokens 和多层特征更适合空间、OCR、图表和细粒度任务。

分辨率提高会增加 token 数，固定 patch 下大致按面积增长，全局 attention 的 score cell 还会出现平方项。动态 resize、切块、token merging、token pruning 和 OCR 都是在信息与成本之间重新分配预算，不能只根据 token 数宣称能力或安全性。

CLIP、SigLIP、MAE 等预训练目标会塑造不同的视觉表示；projector 只能对齐通道维，不能恢复视觉塔已经丢失的信息。可靠的 VLM 视觉入口必须同时满足预处理、位置、token、projector、上下文、证据和评估契约。

下一章将讨论 VLM 架构，进一步解释视觉 token 如何进入语言模型、connector 如何训练，以及多轮图文对话中哪些内容属于输入证据、哪些内容才是生成目标。
