# 第二章：CLIP 与图文对齐

如果一个商品库有一千万张图片，用户输入“带木质桌面的黑色台灯”，系统怎样找到相关图片？传统分类器需要预先列出所有类别，新的组合一出现就要重新设计标签和分类头。图文对齐模型选择了另一条路线：把图片和自然语言描述分别编码成向量，让查询和候选媒体可以在同一个空间里比较。

CLIP 是这条路线最有影响力的代表。它使用图像编码器和文本编码器组成双塔，通过图文配对的对比学习训练共享的 embedding 空间。训练完成后，文本可以检索图片，图片可以检索文字，类别名称也可以临时写成 prompt，作为 zero-shot 分类的候选描述。

但“向量相似”不能被直接翻译成“模型理解了所有关系”。一个全局向量可能知道图片大致是狗，却分不清“狗追猫”和“猫追狗”；它可能知道页面与发票有关，却读不准小字号金额。理解 CLIP，既要理解它为什么有效，也要知道它把哪些信息压缩掉了。

本章从一条可复现的数据流开始：

~~~text
图像 x_i -> image encoder -> 投影 -> 归一化向量 v_i
文本  y_i -> text encoder  -> 投影 -> 归一化向量 t_i
                                      |
                                      v
                           相似度矩阵与对比损失
                                      |
                         检索、zero-shot 分类、评估
~~~

读者会看到五个层次的区别：

1. 图文是否来自同一个配对样本。
2. 两个向量是否在几何空间中相似。
3. 正确配对能否在候选集合中排到前面。
4. zero-shot 类别判断是否在目标分布上可靠。
5. 经过压缩的全局表示是否保留了任务需要的细粒度证据。

前两层是训练目标，后三层才是应用能力和证据边界。把它们混成一句“CLIP 能看懂图片”，会让模型能力和评估结果都失去可解释性。

## 0. 研究对象、资料层级与边界

### 0.1 本章讨论什么

本章讨论 CLIP 风格的双塔图文对齐模型，重点包括：

1. 为什么自然语言可以作为视觉监督。
2. image encoder、text encoder 和 projection head 如何形成形状契约。
3. L2 normalization、cosine similarity 和 temperature 的关系。
4. N 对图文样本如何形成 N x N 相似度矩阵。
5. image-to-text 与 text-to-image 两个方向的 InfoNCE loss。
6. zero-shot 分类、prompt ensemble 和校准。
7. 图文检索的离线索引、指标、延迟和分母。
8. batch size、分布式负样本、重复样本和 false negative。
9. 数据质量、语言覆盖、快捷特征和版权/隐私边界。
10. CLIP 与 VLM 的分工，以及全局 embedding 的能力上限。

本章不会把 SigLIP、BLIP、细粒度 grounding、向量数据库实现或完整的视觉对话训练展开成专题。它们会在后续章节各自讨论；这里会在需要划分边界的地方说明它们与 CLIP 的关系。

### 0.2 三种证据不能混用

阅读论文或产品文档时，至少把证据分成三层：

| 资料类型 | 可以支持的命题 | 不能自动支持的命题 |
| --- | --- | --- |
| 原始论文 | 论文中提出的结构、训练目标和实验设置 | 当前产品的私有数据、价格和线上质量 |
| 官方代码或模型卡 | 某个版本的 API、权重格式和公开实现细节 | 所有部署版本都完全相同 |
| 本章的教学实验 | 公式、形状和指标实现是否自洽 | 真实数据上的泛化能力和安全率 |

CLIP 原论文报告了从互联网收集的大规模图文对，并展示了 zero-shot transfer。这个事实支持“自然语言弱监督可以训练出可迁移的视觉表示”，不支持把任意一个当前服务的训练数据规模、过滤规则或商业质量写成论文结论。后文引用 OpenAI 的公开代码时，也只把它当作某一开源实现的行为证据。

### 0.3 初学者与工程师要分别问什么

初学者可以先问：“匹配的图片和文字，为什么会在向量空间里靠近？”答案是，训练目标把正确文本当作分类目标，使正配对的分数相对更高。

工程师还要继续问：

1. 正配对是否真的相关，还是网页标题噪声。
2. 一个 batch 里有多少独立的负样本。
3. 非对角线样本是否包含实际同义描述。
4. 训练和检索是否使用同一 tokenizer、预处理和归一化。
5. 评估的相关性标注是否覆盖多种正确描述。
6. 线上延迟来自编码、索引还是重排。
7. 目标语言、分辨率、长尾类别是否与训练分布不同。

这些问题不是额外的“优化清单”，而是决定相似度究竟代表什么的建模假设。

## 1. 从固定标签到自然语言监督

### 1.1 固定分类头的能力边界

设传统分类器有 K 个固定类别，图像编码器输出 h，分类头输出：

~~~math
z=Wh+b,\qquad
\hat y=\operatorname{argmax}_{k\in\{1,\ldots,K\}}z_k.
~~~

这种模型适合类别稳定、标注清晰的任务。问题在于，K 个类别是训练时就决定的。模型没有一个自然的机制去回答“这张图片是否符合一个训练时没有出现过的文字描述”。把新类别加入分类头也不是简单地增加一个字符串，因为新头需要有标注样本和重新校准的决策边界。

图文检索的候选集合更开放。用户可能查询：

~~~text
一盏放在木桌上的黑色台灯
一张显示总金额和税额的发票
一只正在追逐另一只猫的狗
~~~

这些查询不是固定的类别 id，而是自然语言组合。若模型把图片和文字映射到共同空间，候选描述可以在推理时产生，分类头也可以由文本向量临时构造。

### 1.2 图文配对提供了什么监督

一条训练样本可以写成：

~~~text
(image_i, caption_i)
~~~

配对关系告诉模型：caption_i 是描述 image_i 的一个文本。它没有告诉模型 caption 中每个词对应哪个像素区域，也没有保证文本完整描述了图片里的所有对象。于是这种监督有两个特点：

1. 它比逐区域标注便宜，能够覆盖大量开放概念。
2. 它比细粒度标注弱，容易携带噪声、偏见和快捷相关性。

例如，网页标题“夏日旅行”可能对应一张海滩照片，但它没有说明照片里有几个人、谁在游泳、太阳位于哪个方向。对比学习会把“夏日旅行”与整张照片联系起来，却不自动得到词与区域的精确对应。

### 1.3 CLIP 的关键变化

CLIP 的关键变化不是单独发明一个更大的 image encoder，而是把训练任务改写成配对识别：

~~~text
给定一张图片和 N 段候选文字，哪一段文字属于这张图片？
给定一段文字和 N 张候选图片，哪一张图片属于这段文字？
~~~

这让自然语言成为开放的监督接口。模型学习到的不是一个固定的 K 类分类头，而是图像和文本之间的可比较几何关系。

需要注意，“开放词表”不等于“任意新概念都可靠”。如果一个概念在训练数据中极少、目标图片与网络图片分布差异大，或 prompt 的语言形式不自然，zero-shot 结果仍可能很差。开放的候选词表扩大了接口，不会取消泛化问题。

### 1.4 一个正配对的最小例子

假设 batch 中有三对样本：

~~~text
image_0: 草地上奔跑的金毛犬
text_0: a golden retriever running on grass

image_1: 一张扫描发票
text_1: a scanned invoice with a total amount

image_2: 一张折线图
text_2: a line chart showing monthly sales
~~~

训练并不要求模型直接生成 text_0，也不要求它先预测 dog、invoice、chart 三个固定类别。它只要求在这个 batch 中：

1. image_0 与 text_0 的分数高于 image_0 与 text_1、text_2。
2. image_1 与 text_1 的分数高于其他文本。
3. image_2 与 text_2 的分数高于其他文本。
4. 反方向也成立：text_0 应优先找到 image_0。

这就是 batch 内负样本的来源，也是后面 false negative 问题的来源。

## 2. 双塔结构与表示契约

### 2.1 数据流

CLIP 的基本结构是两个编码塔和两个投影：

~~~text
image x_i
    -> image encoder f_I
    -> image feature a_i
    -> image projection W_I
    -> v_i

text y_i
    -> tokenizer
    -> text encoder f_T
    -> text feature b_i
    -> text projection W_T
    -> t_i
~~~

图像塔和文本塔在相似度计算之前互不读取对方的 token。它们只在共享空间中通过点积发生交互。这种结构使图库可以提前编码，也使文本库可以提前编码，是检索系统能够扩展到大规模候选集合的主要原因。

### 2.2 形状契约

设 batch size 为 B，图像编码器输出维度为 d_I，文本编码器输出维度为 d_T，共享 embedding 维度为 d：

~~~math
A\in\mathbb{R}^{B\times d_I},\qquad
B_T\in\mathbb{R}^{B\times d_T}.
~~~

投影后：

~~~math
V=A W_I+b_I,\qquad
T=B_T W_T+b_T,
~~~

其中：

~~~math
V,T\in\mathbb{R}^{B\times d}.
~~~

这里用 B_T 表示文本特征矩阵，避免与 batch size 的符号 B 混淆。工程实现中也可以把投影写成线性层或更复杂的 projection head，但最后参与相似度的两个矩阵必须有相同的最后一维。

形状检查至少要回答：

1. 图像和文本 batch 是否一一对应。
2. 两边的 batch size 是否相同。
3. 共享维度 d 是否相同。
4. 归一化是否沿最后一维执行。
5. 相似度矩阵是否为 [B, B]。

如果输出是 [B, d_I] 和 [B, d_T]，却直接相乘，问题不是“模型学得不好”，而是表示契约尚未满足。

### 2.3 Image encoder 不等于共享空间

Image encoder 可以是卷积网络，也可以是 Vision Transformer。它先把像素变成中间表示，再由 projection 把中间维度映射到图文共享空间：

~~~math
a_i=f_I(x_i),\qquad
v_i=\operatorname{normalize}(W_Ia_i).
~~~

这两个向量承担不同责任：

1. a_i 是视觉主干的内部表示，可能保留多层空间信息。
2. v_i 是为跨模态相似度训练过的全局表示。

不能因为 v_i 与文字对齐，就认为 a_i 的每个位置都具有可直接解释的词语含义。区域级 grounding 需要额外的标注、局部特征或交互结构。

### 2.4 Text encoder 不是生成式语言模型

文本塔的目标是把整段文字压缩成一个向量，而不是逐 token 预测下一个词：

~~~math
b_i=f_T(\operatorname{tokenize}(y_i)),\qquad
t_i=\operatorname{normalize}(W_Tb_i).
~~~

它通常使用 Transformer，但“使用 Transformer”并不意味着它具备对话式生成能力。生成模型需要一个逐步输出 token 的解码过程；CLIP text encoder 只需要产生适合相似度比较的句子级表示。

不同实现可能选择结束 token、池化结果或其他聚合方式。使用公开模型时必须读取对应实现的预处理和 pooling 规则，不能只凭“都是 Transformer”来替换 tokenizer 或特殊 token。

### 2.5 L2 归一化与 cosine similarity

对非零向量 z 做 L2 归一化：

~~~math
\operatorname{norm}(z)=\frac{z}{\lVert z\rVert_2},\qquad
\lVert z\rVert_2=\sqrt{\sum_{k=1}^{d}z_k^2}.
~~~

对归一化后的 v_i 和 t_j：

~~~math
v_i^\top t_j
=\frac{(W_Ia_i)^\top(W_Tb_j)}
{\lVert W_Ia_i\rVert_2\lVert W_Tb_j\rVert_2}
=\cos\theta_{ij}.
~~~

因此向量长度不再直接决定相似度，角度成为主要因素。代码中归一化必须处理零向量和数值精度；实际模型输出通常不会是零向量，但教学代码仍应明确异常行为。

这里的“非零”还不够：向量必须是有限数，不能含有 `NaN` 或无穷大。否则归一化后的每个分量都可能变成非有限值，后续 cosine、softmax 和检索排名会一起失去意义。工程接口应在进入共享空间前拒绝这种输入，而不是等到最后的 Recall 变成一个无法解释的数字。

## 3. 相似度矩阵：把配对问题变成分类问题

### 3.1 矩阵构造

将归一化后的图像向量按行堆叠为 V，将文本向量按行堆叠为 T：

~~~math
V\in\mathbb{R}^{B\times d},\qquad
T\in\mathbb{R}^{B\times d}.
~~~

未加温度的 cosine similarity 矩阵为：

~~~math
C=VT^\top,\qquad
C\in\mathbb{R}^{B\times B}.
~~~

元素 C_ij 表示第 i 张图和第 j 段文本的相似度。若 batch 的配对关系确实按相同索引排列，正确配对在对角线：

~~~math
\text{target}_i=i.
~~~

一个三样本矩阵可以示意为：

~~~text
                 text_0   text_1   text_2
image_0            0.91     0.22     0.18
image_1            0.13     0.88     0.31
image_2            0.26     0.35     0.94
~~~

目标不是让所有对角线绝对等于 1，而是让每行的正确列相对其他列更大，同时让每列的正确行相对其他行更大。

### 3.2 为什么矩阵的两个方向都重要

只优化 image-to-text，模型可以学习“给图片找文字”的排序；只优化 text-to-image，模型可以学习相反方向的排序。真实检索经常同时需要两种查询，因此通常对两个方向都计算损失。

两个方向共享同一个 C，但归一化的 softmax 维度不同：

1. image-to-text 对每一行归一化。
2. text-to-image 对每一列归一化，等价于对 C 的转置逐行归一化。

这不是简单的重复计算。一个矩阵可能每行的正样本最强，却有一列被许多图片同时错误地指向；双向目标会暴露这种不对称性。

### 3.3 Batch 内负样本的含义

对 image_i 来说，text_j（j 不等于 i）是当前目标中的负样本。它们只是“当前 batch 里没有与 image_i 配对”的文本，不一定在现实世界中语义不相关。

例如：

~~~text
image_0: 一只金毛犬
text_0: a golden retriever
text_1: a dog running outdoors
~~~

text_1 可能同样适合 image_0。如果把它强行当作负样本，模型会被要求降低一个合理匹配的分数。这种样本称为 false negative。它在大规模互联网数据中很难完全避免，尤其是重复图片、同一事件的多条描述和同一商品的多个角度。

### 3.4 相似度矩阵不是证据解释

C_ij 较大只能说明两个全局表示在训练得到的几何空间里接近。它不能直接说明：

1. 哪个区域导致了分数。
2. 文本中的哪个关系被验证。
3. 图片中的小字是否被正确读取。
4. 模型是否知道一个否定词改变了命题。

如果业务需要金额、日期、坐标或表格单元格，应该引入 OCR、区域裁剪、grounding 或可验证的结构化抽取，不要把一个高 cosine 分数当成完整证据。

## 4. InfoNCE：双向对比损失

### 4.1 Temperature 缩放

令可学习的温度为 tau，logits 为：

~~~math
S_{ij}=\frac{C_{ij}}{\tau}.
~~~

在很多实现中，直接维护 s=log(1/tau)，然后使用：

~~~math
\alpha=\exp(s)=\frac{1}{\tau},\qquad
S=\alpha C.
~~~

这样可以保证缩放因子为正。OpenAI 的公开实现使用 logit_scale 的指数形式，并以与 1/0.07 对应的初始值开始；不同实现可能增加上界或使用其他参数化，部署时要以具体版本为准。

### 4.2 Image-to-text loss

给定图像 i，所有文本是候选类别，正确类别是 i：

~~~math
L_{I\rightarrow T}
=-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{\exp(S_{ii})}
{\sum_{j=1}^{B}\exp(S_{ij})}.
~~~

如果正确分数远大于其他分数，分式接近 1，损失接近 0。如果某个错误文本分数更高，损失会变大。

### 4.3 Text-to-image loss

反方向把每一列视为候选图像：

~~~math
L_{T\rightarrow I}
=-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{\exp(S_{ii})}
{\sum_{j=1}^{B}\exp(S_{ji})}.
~~~

最终通常取平均：

~~~math
L_{\mathrm{align}}
=\frac{1}{2}
\left(
L_{I\rightarrow T}+L_{T\rightarrow I}
\right).
~~~

这个目标看起来像分类，但类别不是固定的 dog、invoice 或 chart，而是当前 batch 中的文本或图片。下一批数据变化时，候选类别也变化。

### 4.4 为什么必须使用稳定的 log-sum-exp

直接计算 exp(S_ij) 可能溢出。对一行 logits z，令 m=max_j z_j：

~~~math
\log\sum_j\exp(z_j)
=m+\log\sum_j\exp(z_j-m).
~~~

由于 z_j-m 小于等于 0，指数项不会因为一个大正数而溢出。交叉熵实现通常内部使用这个技巧；自己实现教学代码时也应该使用稳定版本，而不是仅在小数字例子中调用 exp。

### 4.5 从损失看训练信号

对 image-to-text 的一行，softmax 概率为：

~~~math
p_{ij}=\frac{\exp(S_{ij})}{\sum_k\exp(S_{ik})}.
~~~

对 logits 的梯度有一个很有用的直觉：

~~~math
\frac{\partial L_i}{\partial S_{ij}}
=p_{ij}-\mathbf{1}[j=i].
~~~

因此：

1. 正样本的梯度是 p_ii-1，推动 S_ii 增大。
2. 负样本的梯度是 p_ij，推动 S_ij 减小。
3. 已经很容易区分的负样本概率接近 0，获得的梯度很小。
4. 分数接近的 hard negative 获得更明显的更新。

这解释了为什么数据中的难例有价值，也解释了为什么错误标注的 hard negative 会造成更强的伤害。

## 5. Temperature、批大小与分布式负样本

### 5.1 Temperature 改变什么

假设一行 cosine 分数为：

~~~text
[0.90, 0.50, 0.20]
~~~

当 tau=0.5 时，缩放后的 logits 是 [1.8, 1.0, 0.4]；当 tau=0.05 时，logits 是 [18, 10, 4]。后者的 softmax 更尖锐，模型更强地要求正样本胜出。

较小的 tau 有两个效果：

1. 拉大相似度差异对应的概率差。
2. 让 hard negative 的梯度更集中。

但如果温度过小，softmax 可能迅速饱和，错误标签和异常样本会产生很大的更新；如果温度过大，正负样本差距不够，训练信号变弱。温度不是“越小越好”的质量开关，而是损失几何的一部分。

### 5.2 logit_scale 的参数化

若训练参数是 s，缩放因子为：

~~~math
\alpha=\exp(s).
~~~

优化 s 比直接优化 tau 更方便，因为无需在每一步手工保证 tau 为正。然而 alpha 可能增长过大。工程实现可以：

1. 对 s 或 alpha 设置上界。
2. 记录训练过程中的 temperature 和 logit_scale。
3. 检查验证集的检索指标是否与训练 loss 同步。
4. 将异常样本、重复样本和损失变化一起观察。

如果只看训练 loss 而不记录温度，可能把“缩放变尖”误判为表示质量提升。

### 5.3 Batch size 不只是吞吐参数

在一对一配对假设下，每个图像拥有 B-1 个候选负文本。增大 batch 会增加候选数量，但也会增加：

1. 相似度矩阵的存储，规模约为 B^2。
2. image/text encoder 的激活和显存。
3. 多卡训练时的通信量。
4. false negative 出现的机会。

因此“更大的 batch 一定更好”不成立。更大的 batch 提供更多负样本，但如果数据重复严重，新增的负样本可能只是同义或近重复样本。

### 5.4 多卡训练中的全局负样本

假设有 R 个设备，每个设备有 b 个本地配对。只用本地样本时，每个设备的候选数是 b；将特征跨设备聚合后，候选数可以达到：

~~~math
B_{\mathrm{global}}=R b.
~~~

如果使用 all-gather，标签也要带上设备偏移。rank r 上第 k 个本地样本，在全局矩阵中的目标位置通常是：

~~~math
\operatorname{target}_{r,k}=rb+k.
~~~

如果忘记这个偏移，loss 仍可能运行，却会把正确配对指向错误的全局列。

还要区分两种实现：

1. 只把其他设备特征当作常量，通信和梯度路径较简单，但跨设备特征的梯度行为与完整全局 batch 不同。
2. 使用带梯度的 all-gather，让损失对所有设备的特征反传，语义更接近真正的大 batch，但通信和实现复杂度更高。

具体选择取决于框架版本和目标训练语义，不能只根据代码里出现了 all-gather 就断言等价。

### 5.5 累积梯度不能自动制造更多负样本

梯度累积把多个 micro-batch 的梯度相加，再更新一次参数。若每个 micro-batch 单独计算相似度矩阵，它的负样本仍然只来自该 micro-batch。要得到更大的对比候选集合，需要显式缓存或聚合特征，并处理特征对应的参数版本。

用旧参数编码的缓存特征可能降低通信成本，但会引入 stale negatives。它是否值得，要通过同一验证集上的 Recall@K、训练稳定性和吞吐共同判断。

## 6. 多正样本、重复数据与对比目标的假设

### 6.1 一对一标签只是简化

最基本的 CLIP loss 假设第 i 个图片只对应第 i 个文本。真实数据经常是一对多或多对多：

~~~text
同一张商品图
    -> black desk lamp
    -> a modern lamp on a table
    -> adjustable reading light
~~~

如果这些文本同时出现在 batch 中，一对一 loss 会把其中两个合理文本标记为负样本。此时对角线不是完整的正样本集合。

### 6.2 多正样本的一个改写

令 P_i 是与图像 i 相关的文本索引集合，image-to-text 的多正样本损失可以写成：

~~~math
L_i^{\mathrm{multi+}}
=-\log
\frac{\sum_{j\in P_i}\exp(S_{ij})}
{\sum_{j=1}^{B}\exp(S_{ij})}.
~~~

这个形式允许多个正确候选共同承担分子。实际方法还可能对正样本平均、加权或采用专门的去重策略。重点不是固定某一个公式，而是先承认标注关系可能不是 permutation matrix。

### 6.3 重复和近重复样本

数据去重至少要考虑：

1. 完全相同的图片文件。
2. 经过裁剪、压缩或加水印的近重复图片。
3. 同一网页的多种 caption。
4. 同一事件的多张图片和相同新闻文字。
5. 训练集和验证集之间的图片或文本泄漏。

重复样本会让训练指标看起来很好，也会使检索 Recall 被高估。如果测试集中的“另一个正确描述”被标成错误，模型的实际能力反而可能被低估。评估必须明确相关性集合，而不是默认只有一个字符串正确。

### 6.4 Hard negative 不是越难越好

一个 hard negative 与 query 很像但标签确实不同，例如：

~~~text
正样本：红色无盖水杯
难负样本：蓝色无盖水杯
~~~

它可以帮助模型学习颜色差异。可是，如果标签只来自标题，无法确定图片真的不同，hard negative 也可能是误标。建立 hard negative 时应记录：

1. 负样本的来源。
2. 人工或规则验证的字段。
3. 是否共享同一对象、事件或页面。
4. 该负样本是否只对某个任务成立。

否则，模型会被迫学习数据标注者的偶然区别。

## 7. Zero-shot 图像分类

### 7.1 从文本构造分类器

给定 C 个类别描述 q_1,...,q_C，先用文本塔编码：

~~~math
t_c=\operatorname{normalize}(f_T(q_c)).
~~~

给定一张图片 x，得到：

~~~math
v=\operatorname{normalize}(f_I(x)).
~~~

类别分数为：

~~~math
s_c=\frac{v^\top t_c}{\tau}.
~~~

预测类别：

~~~math
\hat c=\operatorname{argmax}_{c}s_c.
~~~

这里没有重新训练图片分类头。文本向量充当了分类器的候选权重，因此类别名称和 prompt 的写法会影响结果。

### 7.2 Prompt 为什么会改变结果

下列文本指向的语言分布并不相同：

~~~text
dog
a photo of a dog
an image of a dog
a close-up photograph of a golden retriever
~~~

训练数据中的图片标题可能更接近其中某一种写法。prompt 改变了 t_c，也就改变了与图像向量的角度。

prompt ensemble 的常见做法是为一个类别准备 M 个模板，分别得到 t_c,m，先平均再归一化：

~~~math
\bar t_c=
\operatorname{normalize}
\left(
\frac{1}{M}\sum_{m=1}^{M}t_{c,m}
\right).
~~~

平均前是否归一化、平均后是否再次归一化、模板权重是否相同，都属于实现选择。为了让结果可复现，评估报告应保存模板列表和语言版本。

### 7.3 类别名称不是中性的

类别文本还可能包含：

1. 粗粒度和细粒度的层级关系。
2. 同义词、缩写和品牌词。
3. 单数、复数和语言形态。
4. 视觉属性与功能属性。
5. 训练数据中的社会偏见。

例如把“医生”写成一个人物职业类别，和把“穿白大褂的人”写成外观描述，测量的就不是同一个任务。zero-shot 分类的类别定义必须与标注规则一致。

### 7.4 多标签任务不能直接套 softmax

如果一张图片可以同时属于 dog、outdoor 和 pet，多类别互斥 softmax 会强迫模型在这些候选中只选一个。此时可以分别观察每个文本分数，或使用经过校准的阈值：

~~~math
\hat y_c=\mathbf{1}[s_c\ge\gamma_c].
~~~

阈值 gamma_c 不一定对所有类别相同。类别频率、prompt 长度和分布差异都会影响分数，不能默认使用一个未经验证的 0.5。

### 7.5 Zero-shot 的正确评估方式

至少要报告：

1. 类别集合和层级定义。
2. prompt 模板和语言。
3. top-1、top-5 或多标签的具体指标。
4. 每个类别的样本数。
5. 低分辨率、遮挡、长尾和跨域切片。
6. 与监督分类器或简单图像检索基线的比较。

总体准确率很高，可能只是头部类别占比高。若任务涉及医疗、金融或安全，zero-shot 分数不能替代领域标注和人工复核。

## 8. 图文检索：从相似度到可用系统

### 8.1 Text-to-image 的两阶段流程

一个常见的 text-to-image 检索系统有离线和在线两部分。

离线阶段：

1. 读取图片并执行固定的 resize、crop、颜色和归一化。
2. 通过 image encoder 得到 embedding。
3. 对 embedding 做 L2 normalization。
4. 写入向量索引，同时保存图片 id、版本和权限。

在线阶段：

1. 使用与训练一致的 tokenizer 处理 query。
2. 通过 text encoder 得到并归一化 query embedding。
3. 在图片索引中执行 inner product 或 cosine search。
4. 对候选进行权限过滤、去重和可选的重排。
5. 返回图片及其来源信息。

如果离线向量没有归一化而在线 query 已归一化，inner product 不再等价于 cosine。一次预处理版本变更也可能使旧索引和新 query 不在同一个空间。

### 8.2 Image-to-text 不是简单的反向显示

图片检索文字时，文字库可能包含多个层级：

1. 商品标题。
2. alt text。
3. 人工描述。
4. OCR 片段。
5. 文章段落或事件摘要。

不同文本粒度对应不同的相关性。一个准确的短标题可能比一段冗长文章更适合 top-1，但文章中的证据更适合后续问答。评估时要先定义返回对象，而不是只报告一个混合 Recall。

### 8.3 Rank、Recall@K 与 MRR

设 query i 的相关候选集合为 G_i，预测排序为 pi_i(1), pi_i(2), ...。若只有一个相关候选 g_i，target rank 可以写成：

~~~math
r_i=1+\sum_{j\ne i}\mathbf{1}[S_{ij}>S_{ii}].
~~~

但多正样本时，应该取第一个相关结果的 rank：

~~~math
r_i^{+}=\min\{k:\pi_i(k)\in G_i\}.
~~~

Recall@K：

~~~math
\operatorname{Recall@K}
=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[r_i^{+}\le K].
~~~

MRR：

~~~math
\operatorname{MRR}
=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i^{+}}.
~~~

如果查询没有标注任何相关候选，不能把它悄悄计为失败或成功；应定义过滤规则或单独报告 coverage。

排名还需要一个并列规则。教学代码使用“分数降序、原索引升序”的稳定排序，因此完全相同的分数也会得到确定的顺序；这只是 tie-breaking 约定，不是模型真的区分了两个候选。生产评估可以采用平均排名、最差排名或把并列视为同一名次，但必须在报告中固定口径。`N=0`、`K\le0` 或没有任何有效相关集合时，Recall@K 和 MRR 都不应返回一个伪造的均值。

### 8.4 nDCG 与分级相关性

有些检索结果不是只有相关/不相关两种状态。比如同一型号的官方图片、同类商品、背景相似的图片，相关程度不同。设排序位置 k 的相关等级为 rel_k，常见折损累计收益为：

~~~math
\operatorname{DCG@K}
=\sum_{k=1}^{K}
\frac{2^{rel_k}-1}{\log_2(k+1)}.
~~~

用理想排序的 DCG 归一化得到 nDCG。它适合分级标注，但仍依赖标注者一致性。指标变复杂不等于证据自动变强，最重要的是保存相关性定义和标注分母。

### 8.5 Exact search、ANN 与重排

候选库较小时，可以对所有归一化向量做精确矩阵乘法。候选规模很大时，常使用 approximate nearest neighbor index。近似索引提供速度和规模，但可能牺牲召回。

生产系统常分为：

~~~text
query -> embedding -> ANN top-K' -> permission/filter/dedup
      -> optional reranker -> final top-K
~~~

若最终返回 K 个结果，ANN 可以先取 K' 大于 K 的候选，为过滤和去重留出余量。应同时记录：

1. embedding 计算延迟。
2. ANN 查询延迟。
3. 过滤和重排延迟。
4. 候选 K'、最终 K。
5. exact search 与 ANN 的 Recall 差异。

只报告总 QPS，无法判断模型、索引或业务过滤哪一层成为瓶颈。

## 9. 训练数据：对齐质量从网页之外开始

### 9.1 Caption 噪声

图文对可能来自标题、alt text、文件名、标签或正文。常见噪声包括：

1. 文字只描述网页主题，不描述图片。
2. 多张图片共享同一个标题。
3. caption 含有模板、广告或导航文字。
4. 图片是拼图、截图或重复转载。
5. 语言、编码和标点混杂。
6. 文本描述带有社会偏见或隐含身份推断。

模型的 embedding 会吸收训练目标中最稳定的相关性。若图片总带有某种水印，模型可能把水印当成类别线索；若某个职业在数据中经常与特定性别图像共现，zero-shot 结果可能反映数据偏差而不是视觉事实。

### 9.2 过滤与去重是建模的一部分

过滤规则会改变模型学到的概念边界。工程记录应包括：

1. 原始配对数量和过滤后数量。
2. 各语言、域和类别的保留率。
3. 图片和文本的去重方法。
4. 训练、验证和测试的拆分依据。
5. 敏感内容、版权和删除请求的处理。
6. 过滤器的误删和漏删抽样。

“清洗后数据更干净”不是充分说明。清洗可能删除低资源语言、少数场景和真实长尾，造成分布变窄。

### 9.3 语言和领域迁移

CLIP 的共同空间不是天然语言无关。文本塔对不同语言、脚本和短语形式的表示质量可能不同；图片塔对互联网照片、扫描文档、工业相机和医学影像的分布也不同。

评估应把以下切片分开：

1. 语言。
2. 图片来源和分辨率。
3. 文本长度。
4. OCR 密度。
5. 类别频率。
6. 组合关系和否定表达。

如果中文 query 的 Recall 下降，原因可能是 tokenizer、训练数据、prompt 模板或索引数据，而不应笼统称为“中文理解差”。

### 9.4 数据许可和隐私

图像和 caption 可能含有人脸、车牌、合同、医疗信息和受版权保护内容。训练和部署至少需要明确：

1. 数据的来源与许可范围。
2. 是否允许训练、索引和用户查询。
3. 删除请求如何影响原图、embedding、缓存和备份。
4. 租户之间是否隔离索引和访问权限。
5. 返回结果是否暴露超出用户权限的图片。

embedding 不是天然匿名化。即使不能还原原图，向量也可能成为敏感数据资产，应按系统的威胁模型保护。

## 10. CLIP 的工程实现

### 10.1 最小模型接口

一个可替换的 CLIP 风格模型，接口可以抽象为：

~~~text
encode_image(images) -> [B, d]
encode_text(input_ids, attention_mask) -> [B, d]
similarity(image_features, text_features) -> [B, B]
loss(image_features, text_features, pair_ids) -> scalar
~~~

接口中还应显式保存：

1. image preprocessing 版本。
2. tokenizer 和 special token 版本。
3. projection 维度。
4. normalization 位置。
5. temperature 或 logit_scale。
6. 图文配对的唯一 id。

如果模型服务只返回一个向量而不返回这些元数据，后续索引迁移和结果复现会变得困难。

### 10.2 PyTorch 训练骨架的责任边界

下面的代码只展示损失层。image_encoder 和 text_encoder 的输出必须已经满足 batch 对齐，具体编码器、padding 和数据加载不在这个函数里：

~~~python
import math
import torch
import torch.nn.functional as F


def symmetric_clip_loss(image_features, text_features, logit_scale):
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("features must have shape [batch, dim]")
    if image_features.shape != text_features.shape:
        raise ValueError("paired features must have the same shape")
    batch_size, feature_dim = image_features.shape
    if batch_size == 0 or feature_dim == 0:
        raise ValueError("features must have a non-empty batch and dimension")
    if not image_features.is_floating_point() or not text_features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")
    if image_features.device != text_features.device:
        raise ValueError("image and text features must be on the same device")
    if not torch.isfinite(image_features).all().item() or not torch.isfinite(
        text_features
    ).all().item():
        raise ValueError("features must contain only finite values")
    if (image_features.norm(dim=-1) == 0).any().item() or (
        text_features.norm(dim=-1) == 0
    ).any().item():
        raise ValueError("zero feature vectors cannot be normalized")
    if logit_scale.ndim != 0 or not logit_scale.is_floating_point():
        raise ValueError("logit_scale must be a floating-point scalar")
    if logit_scale.device != image_features.device:
        raise ValueError("logit_scale and features must be on the same device")
    if not torch.isfinite(logit_scale).item():
        raise ValueError("logit_scale must be finite")
    if logit_scale.item() > math.log(torch.finfo(logit_scale.dtype).max):
        raise ValueError("logit_scale.exp() would overflow")

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)

    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.T, labels)
    return (image_to_text + text_to_image) / 2
~~~

这段代码没有处理多正样本、分布式 all-gather、权限过滤或混合精度。它的价值是把最小的 shape contract 和 loss 责任说清楚，而不是冒充完整训练系统。

### 10.3 分布式实现的审计点

多卡训练时，审计记录至少包含：

1. local batch size 和 global batch size。
2. 是否聚合图像和文本特征。
3. label offset 是否正确。
4. all-gather 是否允许梯度。
5. 每次 optimizer step 使用了多少有效配对。
6. 重复样本是否跨 rank 出现。
7. 温度参数是共享、复制还是单独更新。

如果只在 rank 0 打印 local loss，它不一定代表全局 loss。应同时报告 global pair count、有效样本数和聚合方式。

### 10.4 数值精度与归一化顺序

相似度通常在半精度中也可以计算，但要留意：

1. 向量 norm 的累积误差。
2. 很大的 logit_scale 与 exp 溢出。
3. softmax 的稳定实现。
4. all-gather 后的 dtype 一致性。
5. 索引构建时是否重新归一化。

一个稳妥的验证顺序是先用 float32 运行小 batch，检查 loss、梯度和排名，再切换 autocast 或低精度。不要在 shape 和标签尚未验证时同时引入量化、近似索引和多卡通信。

## 11. CLIP 与 VLM：共享视觉基础，不是同一种任务

### 11.1 CLIP 输出什么

CLIP 的标准输出是图像 embedding、文本 embedding 和它们的相似度。它回答的是：

~~~text
这张图片和这些文字，哪一组更匹配？
~~~

它通常不会直接生成一段新的 assistant 文本，也不会自动输出证据区域。

### 11.2 VLM 需要额外的交互机制

一个常见 VLM 数据流是：

~~~text
image -> vision encoder -> visual features -> connector -> language model
text  -> tokenizer --------------------------------------^
~~~

语言模型可以逐 token 生成回答，但它是否使用了正确的视觉证据，取决于视觉 token 数、connector、训练数据和评估。CLIP 的 vision tower 可以作为视觉编码器的一种来源，但把 CLIP 接到 LLM 上并不会自动获得 OCR、计数、空间 grounding 或工具安全能力。

### 11.3 全局向量和局部 token 的取舍

CLIP 的全局 embedding 适合：

1. 粗粒度检索。
2. 开放词表分类。
3. 图片和文本的快速匹配。
4. 作为候选召回阶段。

局部 token 或区域特征更适合：

1. 读取小字号文字。
2. 判断多个对象的相对位置。
3. 返回 bounding box 或时间片段。
4. 处理表格和版面。

更多 token 会增加上下文和计算成本。一个常见系统会先用 CLIP 做便宜召回，再用区域模型、OCR 或 VLM 对少量候选做精确处理。

### 11.4 对齐不等于指令遵循

图文对比学习优化的是匹配排序；instruction tuning 优化的是在用户条件下生成合适的响应。一个模型可以有很好的图文检索 Recall，却不能回答“图中左边第二个物体是什么”，也不能安全地执行截图里的工具指令。

因此评估表中要把这些指标分开：

| 能力 | 典型输出 | 不能由哪项指标替代 |
| --- | --- | --- |
| 全局对齐 | cosine、Recall@K | 不能替代区域 grounding |
| zero-shot 分类 | top-1、macro-F1 | 不能替代生成质量 |
| OCR/文档理解 | 字段和坐标 | 不能替代全局检索 |
| 对话和指令 | 回答、工具参数 | 不能替代相似度 |

## 12. 失败模式与反事实评估

### 12.1 词汇相似但关系相反

考虑两个文本：

~~~text
a dog chasing a cat
a cat chasing a dog
~~~

它们共享 dog、cat、chasing 等词，文本向量可能很接近。若任务要求区分主客体关系，只看全局 cosine 可能不够。

一个反事实测试可以固定图片，替换 query 中的主客体；或固定 query，交换图片中对象的关系。若排名几乎不变，模型可能依赖词汇共现而没有可靠编码关系。

### 12.2 计数与 OCR

“两只杯子”和“一只杯子”需要数量信息；“总金额为 1280 元”需要字符和版面信息。全局 embedding 为检索主题保留了足够信息，并不代表它能把这些字段精确解码出来。

评估应分别测试：

1. 物体是否出现。
2. 数量是否正确。
3. 属性是否绑定到正确对象。
4. OCR 字符是否正确。
5. 表格行列关系是否保留。

### 12.3 Shortcut 与水印

若训练数据中某类图片经常带有特殊水印，模型可能用水印判断类别。去掉水印、替换背景或裁剪边缘可以构造反事实测试：

~~~text
原图 -> 去水印
原图 -> 换背景
原图 -> 保留主体但改文字
~~~

如果分数大幅变化而语义主体未变，就需要检查数据相关性，而不是立即把结果解释成视觉理解。

### 12.4 Prompt 和语言偏差

同一个类别用中文、英文、缩写和描述句表达，分数可能不同。应该记录：

1. query 语言。
2. prompt 模板。
3. tokenizer 长度。
4. 类别词的频率和歧义。
5. 每种语言的样本量和置信区间。

跨语言平均分可能掩盖低资源语言的严重下降。语言切片不是可选的装饰统计，而是共同空间是否公平可用的证据。

### 12.5 高分不等于授权

在文档检索系统中，CLIP 可以把一张内部合同图片排到高位，但相似度不应决定用户是否有权看到它。向量检索之后必须执行租户、文档和字段权限过滤。媒体中的文字也不应因为与 query 相似就升级成系统指令。

## 13. 贯穿案例：合同图片与条款检索

### 13.1 业务问题

一个企业知识库包含扫描合同、报价单和会议截图。用户输入：

~~~text
找出包含“付款期限超过 60 天”条款的合同页面。
~~~

纯关键词检索可能漏掉“自验收之日起两个月内付款”这种表达；纯 CLIP 检索又可能把“付款期限”主题相似但数值不符合的页面排在前面。

### 13.2 分层系统

一个更可靠的流程是：

~~~text
页面图像
  -> CLIP image embedding -> 主题候选召回
  -> OCR + layout parsing -> 数字、单位、行列和坐标
  -> 文本/规则/小型重排器 -> 60 天条件核验
  -> 权限过滤 -> 带页码和区域引用的结果
~~~

CLIP 在这里承担召回责任，不承担最终金额或期限证明责任。它可以提高候选覆盖，OCR 和规则负责精确字段，权限系统负责用户能否看到结果。

### 13.3 评估分母

应分别记录：

1. 页面召回率：包含目标条款的页面是否进入候选 top-K。
2. 字段准确率：期限数值和单位是否正确。
3. 引用支持率：返回的页码和区域是否真的支持结论。
4. 权限违规次数：不应返回的页面是否被展示。
5. 单位成功成本：完成一次有正确引用的查询消耗的编码、索引和重排资源。

若只报告 CLIP Recall@10，无法知道最终用户得到的“超过 60 天”是否是被文字证据支持的。

### 13.4 失败复盘

一次典型失败可能是：

1. CLIP 正确召回了相关页面。
2. OCR 把“60”识别成“80”。
3. 规则层据此判断条款满足条件。
4. 返回了错误结论，但相似度和召回指标都很好。

这说明图文对齐、文字识别和业务规则各有责任边界。排查时应保存每层输入、输出、版本和区域，而不是只保存最终答案。

## 14. 最小可运行对齐、检索与 zero-shot 审计

下面的程序只使用 Python 标准库。它构造低维的合成图像和文本向量，用同一套数学结构计算：

1. 归一化和 cosine similarity。
2. 稳定的双向对比损失。
3. image-to-text、text-to-image 的排名。
4. Recall@1 和 MRR。
5. prompt ensemble 的 zero-shot 分类。
6. 多正样本损失的一个小例子。

合成向量不是训练出来的 CLIP，也不代表真实数据质量。程序的作用是把索引、温度、标签和分母的关系变成可检查的对象。

~~~python
from math import exp, isfinite, log, sqrt


def normalize(vector):
    if not vector or not all(isfinite(value) for value in vector):
        raise ValueError("vector must be non-empty and finite")
    norm = sqrt(sum(value * value for value in vector))
    if not isfinite(norm) or norm == 0:
        raise ValueError("zero vector cannot be normalized")
    return [value / norm for value in vector]


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return sum(x * y for x, y in zip(left, right))


def logsumexp(values):
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("logits must be non-empty and finite")
    maximum = max(values)
    total = sum(exp(value - maximum) for value in values)
    if not isfinite(total) or total <= 0:
        raise ValueError("logsumexp received an invalid sum")
    return maximum + log(total)


def cross_entropy_row(logits, target):
    if not isinstance(target, int) or target < 0 or target >= len(logits):
        raise ValueError("target must be a valid logit index")
    return logsumexp(logits) - logits[target]


def transpose(matrix):
    if not matrix or any(not row for row in matrix):
        raise ValueError("matrix must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    return [list(row) for row in zip(*matrix)]


def rank_of_target(scores, target):
    if not scores or not all(isfinite(score) for score in scores):
        raise ValueError("scores must be non-empty and finite")
    if not isinstance(target, int) or target < 0 or target >= len(scores):
        raise ValueError("target must be a valid score index")
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    return order.index(target) + 1


def first_positive_rank(scores, positive_indices):
    if not scores or not all(isfinite(score) for score in scores):
        raise ValueError("scores must be non-empty and finite")
    if not positive_indices:
        raise ValueError("positive set cannot be empty")
    if any(
        not isinstance(index, int) or index < 0 or index >= len(scores)
        for index in positive_indices
    ):
        raise ValueError("positive indices must be valid score indices")
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    for rank, index in enumerate(order, start=1):
        if index in positive_indices:
            return rank
    raise RuntimeError("positive set was not found after validation")


def multi_positive_row_loss(logits, positive_indices):
    if not logits or not all(isfinite(value) for value in logits):
        raise ValueError("logits must be non-empty and finite")
    if not positive_indices:
        raise ValueError("positive set cannot be empty")
    if any(
        not isinstance(index, int) or index < 0 or index >= len(logits)
        for index in positive_indices
    ):
        raise ValueError("positive indices must be valid logit indices")
    numerator = logsumexp([logits[index] for index in positive_indices])
    return logsumexp(logits) - numerator


image_names = ["dog_photo", "receipt_photo", "chart_photo"]
text_names = ["a dog on grass", "a scanned receipt", "a line chart"]

image_raw = [
    [0.90, 0.10, 0.00, 0.05],
    [0.10, 0.92, 0.05, 0.00],
    [0.00, 0.10, 0.88, 0.12],
]
text_raw = [
    [0.85, 0.05, 0.02, 0.00],
    [0.05, 0.88, 0.10, 0.02],
    [0.02, 0.08, 0.91, 0.10],
]

image_features = [normalize(vector) for vector in image_raw]
text_features = [normalize(vector) for vector in text_raw]
temperature = 0.07
logit_scale = 1.0 / temperature

similarity = [
    [dot(image, text) for text in text_features]
    for image in image_features
]
logits = [
    [score * logit_scale for score in row]
    for row in similarity
]
labels = list(range(len(logits)))

loss_i2t = sum(
    cross_entropy_row(row, target)
    for row, target in zip(logits, labels)
) / len(labels)
loss_t2i = sum(
    cross_entropy_row(row, target)
    for row, target in zip(transpose(logits), labels)
) / len(labels)
alignment_loss = (loss_i2t + loss_t2i) / 2

ranks_i2t = [
    rank_of_target(row, target)
    for row, target in zip(logits, labels)
]
ranks_t2i = [
    rank_of_target(row, target)
    for row, target in zip(transpose(logits), labels)
]
recall_at_1 = sum(rank == 1 for rank in ranks_i2t) / len(ranks_i2t)
mrr = sum(1 / rank for rank in ranks_i2t) / len(ranks_i2t)

zero_shot_prompts = {
    "dog": [
        [0.86, 0.06, 0.01, 0.00],
        [0.82, 0.10, 0.02, 0.03],
    ],
    "receipt": [
        [0.04, 0.90, 0.12, 0.01],
        [0.08, 0.84, 0.10, 0.04],
    ],
    "chart": [
        [0.02, 0.10, 0.90, 0.12],
        [0.01, 0.12, 0.86, 0.14],
    ],
}

class_vectors = {}
for class_name, variants in zero_shot_prompts.items():
    normalized_variants = [normalize(vector) for vector in variants]
    averaged = [
        sum(values) / len(values)
        for values in zip(*normalized_variants)
    ]
    class_vectors[class_name] = normalize(averaged)

query_image = normalize([0.03, 0.12, 0.90, 0.10])
class_scores = {
    class_name: dot(query_image, vector)
    for class_name, vector in class_vectors.items()
}
predicted_label = max(class_scores, key=class_scores.get)

multi_positive_logits = logits[0]
multi_positive_loss = multi_positive_row_loss(
    multi_positive_logits,
    positive_indices={0, 1},
)

checks = {
    "diagonal_is_best": all(
        rank == 1
        for rank in ranks_i2t + ranks_t2i
    ),
    "loss_is_small": alignment_loss < 0.001,
    "retrieval_is_correct": (
        recall_at_1 == 1.0
        and mrr == 1.0
    ),
    "zero_shot_is_correct": predicted_label == "chart",
    "multi_positive_is_no_larger": (
        multi_positive_loss <= cross_entropy_row(
            multi_positive_logits,
            target=0,
        )
    ),
}

signals = []
actions = []
if not checks["diagonal_is_best"]:
    signals.append("a paired item is not ranked first")
    actions.append("inspect pair ids and preprocessing alignment")
if multi_positive_loss > cross_entropy_row(multi_positive_logits, 0):
    signals.append("multi-positive loss is unexpectedly larger")
    actions.append("inspect positive sets and logsumexp implementation")
if not checks["zero_shot_is_correct"]:
    signals.append("prompt ensemble selects the wrong synthetic class")
    actions.append("inspect class prompts and feature normalization")

decision = (
    "continue_to_real_data_evaluation"
    if all(checks.values())
    else "repair_alignment_contract"
)

print(
    "similarity_matrix=",
    [[round(value, 3) for value in row] for row in similarity],
)
print(
    "losses=",
    {
        "image_to_text": round(loss_i2t, 6),
        "text_to_image": round(loss_t2i, 6),
        "symmetric": round(alignment_loss, 6),
        "multi_positive_row": round(multi_positive_loss, 6),
    },
)
print("ranks_i2t=", dict(zip(image_names, ranks_i2t)))
print("ranks_t2i=", dict(zip(text_names, ranks_t2i)))
print(
    "retrieval_metrics=",
    {
        "recall_at_1": round(recall_at_1, 3),
        "mrr": round(mrr, 3),
    },
)
print(
    "zero_shot_scores=",
    {
        key: round(value, 3)
        for key, value in class_scores.items()
    },
)
print("predicted_label=", predicted_label)
print("signals=", signals)
print("actions=", actions)
print("checks=", checks)
print("decision=", decision)
~~~

这个实验有意把检查结果拆开。对角线排名验证配对索引，loss 验证数值实现，检索指标验证排序，zero-shot 验证 prompt 候选，multi-positive 行验证多个正样本不会被错误地全部放进分母之外。即使所有 checks 都为 True，也只说明合成向量和代码契约正确，不说明真实 CLIP 能读懂合同金额或空间关系。

## 15. 练习：从相似度走到证据

1. 设有四对图文样本，写出 V、T 和 C 的形状，并标出 image-to-text 的标签向量。
2. 构造一个相似度矩阵，使每行的对角线最大但某一列被多个图片错误指向，解释为什么双向 loss 有帮助。
3. 对同一行 logits 分别使用 tau=0.5 和 tau=0.05，计算 softmax 并描述梯度集中位置。
4. 推导稳定 log-sum-exp 与直接 exp 求和在大 logits 下的数值差异。
5. 给定一张图片对应三个合理 caption，写出多正样本损失，并说明它与一对一损失的不同。
6. 设计一个四类别 zero-shot 分类任务，列出每个类别的三个 prompt，并说明类别定义如何影响标签。
7. 计算一组检索结果的 Recall@1、Recall@5 和 MRR；再加入第二个相关结果，说明分母是否变化。
8. 设计一个 exact search 与 ANN search 的对照实验，同时记录 Recall、延迟和索引内存。
9. 为中文、英文和低分辨率图片分别建立评估切片，写出每个切片的样本数和置信区间。
10. 构造“去水印”“换背景”“交换主客体”三个反事实测试，判断模型是否依赖快捷特征。
11. 在多卡场景中，计算四个 rank、每 rank 两个样本时的全局 label offset。
12. 为合同页面检索系统划分 CLIP 召回、OCR 抽取、权限过滤和引用验证四个责任边界，并为每一层设计一个失败指标。

## 16. 资料入口与证据边界

1. Radford et al., Learning Transferable Visual Models From Natural Language，CLIP 原始论文：<https://arxiv.org/abs/2103.00020>。
2. OpenAI CLIP 官方代码与 README：<https://github.com/openai/CLIP>。
3. van den Oord et al., Representation Learning with Contrastive Predictive Coding：<https://arxiv.org/abs/1807.03748>。
4. Jia et al., Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision，ALIGN：<https://arxiv.org/abs/2102.05918>。
5. Zhai et al., LiT: Zero-Shot Transfer with Locked-image text Tuning：<https://arxiv.org/abs/2111.07991>。
6. Zhai et al., Sigmoid Loss for Language Image Pre-Training：<https://arxiv.org/abs/2303.15343>。
7. Johnson et al., Billion-scale similarity search with GPUs，FAISS 论文：<https://arxiv.org/abs/1702.08734>。

CLIP 论文支持双塔、图文对比目标和论文中的迁移实验；官方代码支持特定实现中的 normalization、logit_scale 和接口行为；ALIGN、LiT 和 SigLIP 说明图文对齐路线存在不同的数据规模、初始化和损失设计；FAISS 论文支持大规模相似度搜索的工程背景。这些资料都不能直接证明某个业务数据集上的 Recall、偏差、安全率、吞吐或成本。

当资料声称一个模型“支持开放词表”“理解图片”或“具备强大的跨模态推理能力”时，至少要继续追问：

1. 输入图像分辨率和裁剪是什么。
2. 文本语言和 prompt 是什么。
3. 候选集合与相关性标签是什么。
4. 是否包含近重复或数据泄漏。
5. 是全局检索、局部 grounding 还是生成问答。
6. 报告的是平均值还是包含长尾切片。
7. 结果是否来自公开权重、哪一个版本和哪一套预处理。

## 17. 本章回顾

CLIP 将图像和文字分别编码到共享 embedding 空间，用相似度矩阵把图文配对识别写成 batch 内分类。L2 normalization 让点积对应 cosine similarity，temperature 控制 logits 的锐度，双向 InfoNCE 同时训练 image-to-text 与 text-to-image。

这条路线之所以适合检索，是因为双塔允许图片和文字离线编码；它之所以能支持 zero-shot 分类，是因为类别可以在推理时用 prompt 表示。但它的表示主要是全局的，图文配对监督也没有提供完整区域、数量、关系和 OCR 标签。因此高 Recall、正确 top-1 或较低训练 loss 都不能单独证明细粒度理解。

真正可用的系统还必须处理 batch 与全局负样本、重复和 false negative、温度和数值稳定性、索引版本、权限过滤、语言与领域切片、数据许可、反事实测试和单位成功成本。把这些边界写进数据和评估契约，才是从“会计算相似度”走向“知道相似度能证明什么”的关键。

下一章将进入 vision encoder，解释 patch、位置、层级特征和视觉分辨率如何决定图像信息最终能否进入这个共享空间。
