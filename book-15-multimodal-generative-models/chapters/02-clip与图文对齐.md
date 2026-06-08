# 第二章：CLIP 与图文对齐

CLIP 是多模态领域最重要的基础模型之一。它的核心贡献不是发明了图像分类模型，也不是发明了文本编码器，而是用大规模图文对比学习把图像和文本对齐到同一个语义空间。对齐之后，模型可以做文本检索图片、图片检索文本、zero-shot 图像分类，也可以作为后续 VLM、图像生成和多模态检索系统的重要基础。

如果只用一句话概括 CLIP：它训练一个 image encoder 和一个 text encoder，让匹配的图片和文本向量相似，不匹配的图片和文本向量不相似。

本章目标是讲清 CLIP 的动机、结构、训练目标、相似度矩阵、InfoNCE loss、zero-shot classification、retrieval、工程实现和局限。

## 0. 本讲资料边界与第二轮精修口径

第二轮精修前，本讲按 `WRITING_PLAN.md` 核对了 CLIP 论文、OpenAI CLIP 官方代码和 README 示例、InfoNCE / Contrastive Predictive Coding 论文，以及前一章多模态总览中的 token / connector / 成本审计口径。需要注意：

1. CLIP 的核心不是“把图片分类头换成文本”，而是用大规模图文对学习 image encoder 和 text encoder 的共享 embedding 空间。
2. 论文摘要强调的是从自然语言监督中学习可迁移视觉表示，预训练任务可以理解为“预测哪段 caption 属于哪张图片”，并在 zero-shot transfer 上发挥作用。
3. OpenAI 官方代码中，图像和文本特征会先 L2 normalize，再用 `logit_scale.exp()` 乘以点积得到 `logits_per_image` 和 `logits_per_text`；`logit_scale` 初始值对应 `1 / 0.07`。
4. 官方 README 的 zero-shot 示例把归一化后的图像和文本特征做矩阵乘法，并用 `100.0 * image_features @ text_features.T` 后接 softmax 得到类别概率。
5. 本章只讲 CLIP 风格双塔对齐、InfoNCE、zero-shot 分类和检索评估；不展开 SigLIP、BLIP、LLaVA、VLM connector、细粒度 grounding 或大规模向量数据库工程，这些放到后续章节或百科条目中。

本章第二轮重点是把相似度矩阵、双向 loss、temperature、检索指标和最小可运行 demo 补齐，让读者能从公式、代码和面试表达三个角度讲清 CLIP。

## 2.1 为什么需要图文对齐

传统视觉模型通常依赖固定类别标签。例如 ImageNet 分类模型只能在预定义的 1000 个类别里预测。

这种方式有几个问题：

1. 类别空间固定，无法开放词表识别。
2. 标注成本高。
3. 标签信息太短，不能表达复杂语义。
4. 很难直接支持图文检索。
5. 不能自然和语言模型连接。

互联网中存在大量图文对：图片和标题、图片和 alt text、商品图和描述、新闻图和正文。CLIP 的思路是利用这些天然弱监督数据，让模型从图文配对中学习视觉和语言的对应关系。

例如：

```text
image: 一只金毛犬在草地上奔跑
text: a golden retriever running on the grass
```

模型不需要人工标注类别，只要知道这张图和这段文本是一对，就可以学习语义对齐。

面试回答：

```text
图文对齐的目标是让图像和文本进入同一个语义空间，使语义匹配的图文向量相似，不匹配的图文向量相远。这样模型就不再局限于固定分类标签，可以支持开放词表分类、图文检索和后续视觉语言模型。
```

## 2.2 CLIP 的基本结构

CLIP 是双塔结构：

```text
image -> image encoder -> image embedding
text  -> text encoder  -> text embedding
```

然后计算 image embedding 和 text embedding 的相似度。

典型流程：

1. 一个 batch 里有 `N` 对图文样本。
2. image encoder 编码 `N` 张图片。
3. text encoder 编码 `N` 段文本。
4. 得到 `N` 个 image embeddings 和 `N` 个 text embeddings。
5. 计算 `N x N` 相似度矩阵。
6. 对角线是正确图文对，非对角线是不匹配图文对。

双塔结构的好处：

1. 图像和文本可以离线分别编码。
2. 检索时可以预先建立向量索引。
3. 推理效率高。
4. 适合大规模图文检索。

缺点：

1. 图像和文本只在最后通过相似度交互。
2. 细粒度推理能力不如 cross-attention VLM。
3. 很难直接回答复杂问题。

## 2.3 Image Encoder

CLIP 的 image encoder 可以是 ResNet，也可以是 Vision Transformer。

Vision Transformer 路线中，图片会被切成 patch：

```text
image -> patch embeddings -> Transformer -> image embedding
```

例如一张图片被切成 `16 x 16` patch，每个 patch 类似一个视觉 token。经过视觉 Transformer 后，取 CLS token 或池化结果作为整张图片的 embedding。

输出通常会投影到共享维度：

```text
image_embedding: [N, d]
```

其中 `N` 是 batch size，`d` 是对齐空间维度。

## 2.4 Text Encoder

Text encoder 通常是 Transformer，把文本 token 编码成文本 embedding。

流程：

```text
text -> tokenizer -> token ids -> text transformer -> text embedding
```

输出：

```text
text_embedding: [N, d]
```

注意，CLIP 的 text encoder 不是生成式 LLM，它的目标不是 next-token prediction，而是产生适合和图像对齐的文本向量。

在 zero-shot 分类时，会把类别名称改写成 prompt，例如：

```text
a photo of a dog
a photo of a cat
a photo of a car
```

再用 text encoder 得到类别文本向量。

## 2.5 向量归一化和余弦相似度

CLIP 通常会对图像和文本 embedding 做归一化。

```python
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

归一化后，点积就等价于 cosine similarity：

```python
logits = image_features @ text_features.T
```

Shape：

```text
image_features: [N, d]
text_features: [N, d]
logits: [N, N]
```

`logits[i, j]` 表示第 `i` 张图片和第 `j` 段文本的相似度。

对角线 `logits[i, i]` 是正确配对，非对角线是不匹配配对。

## 2.6 相似度矩阵的直觉

假设 batch 里有 4 对图文：

```text
image_0 <-> text_0
image_1 <-> text_1
image_2 <-> text_2
image_3 <-> text_3
```

相似度矩阵：

```text
          text_0 text_1 text_2 text_3
image_0     *      -      -      -
image_1     -      *      -      -
image_2     -      -      *      -
image_3     -      -      -      *
```

星号位置是正样本，其他位置是 batch 内负样本。

训练目标是让对角线分数高，非对角线分数低。

这就是 in-batch negatives 的思想：一个 batch 内的其他文本可以作为当前图片的负样本，其他图片也可以作为当前文本的负样本。

## 2.7 CLIP Loss：双向对比学习

CLIP 的 loss 通常是双向的：

1. image-to-text：给定图片，预测正确文本。
2. text-to-image：给定文本，预测正确图片。

代码直觉：

```python
import torch
import torch.nn.functional as F


def clip_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

这里：

1. `logits` 是 `[N, N]`。
2. `labels = [0, 1, ..., N-1]`。
3. 对第 `i` 张图来说，正确文本是第 `i` 个。
4. 对第 `i` 段文本来说，正确图片是第 `i` 张。

`logit_scale` 是可学习或固定的温度缩放参数，用于控制相似度分布的尖锐程度。

### 2.7.1 关键公式与 CLIP 速查

设一个 batch 中有 `N` 对图文样本 `(x_i, y_i)`。图像 encoder 和文本 encoder 分别得到未归一化特征：

```math
a_i=f_I(x_i),\qquad b_i=f_T(y_i)
```

投影到共享维度并做 L2 normalize：

```math
v_i=\frac{W_I a_i}{\lVert W_I a_i\rVert_2},\qquad
t_i=\frac{W_T b_i}{\lVert W_T b_i\rVert_2}
```

归一化后，点积就是 cosine similarity。CLIP 相似度矩阵可以写成：

```math
S_{ij}=\frac{v_i^\top t_j}{\tau}
```

其中 `tau` 是 temperature；工程实现中常用 `logit_scale = 1 / tau` 或学习 `logit_scale` 的对数形式。`tau` 越小，softmax 分布越尖锐；`tau` 太大，正负样本区分会变弱。

image-to-text 方向的 loss 是：

```math
L_{I2T}=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(S_{ii})}
{\sum_{j=1}^{N}\exp(S_{ij})}
```

text-to-image 方向对应相似度矩阵转置：

```math
L_{T2I}=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(S_{ii})}
{\sum_{j=1}^{N}\exp(S_{ji})}
```

最终 CLIP loss 通常取两个方向平均：

```math
L_{\mathrm{CLIP}}=\frac{L_{I2T}+L_{T2I}}{2}
```

图文检索时，给定第 `i` 张图片，它的正确文本 rank 可以写成：

```math
r_i=1+\sum_{j\ne i}\mathbf{1}[S_{ij}>S_{ii}]
```

Recall@K 和 MRR 分别是：

```math
R_K=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[r_i\le K]
```

```math
\mathrm{MRR}=
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}
```

zero-shot 分类时，类别 `c` 会被写成一个或多个 prompt。若一个类别有 `M` 个 prompt embedding `t_{c,m}`，prompt ensemble 可以先平均再归一化：

```math
\bar t_c=
\frac{\frac{1}{M}\sum_{m=1}^{M}t_{c,m}}
{\left\lVert\frac{1}{M}\sum_{m=1}^{M}t_{c,m}\right\rVert_2}
```

预测类别为：

```math
\hat c=\arg\max_c v^\top \bar t_c
```

面试时要把这几件事连起来：双塔编码、L2 normalize、`N x N` 相似度矩阵、对角线标签、双向 cross entropy、temperature、zero-shot prompt 和检索指标。

## 2.8 InfoNCE 的直觉

InfoNCE 可以理解为“从多个候选中找出正样本”的分类目标。

对一张图片来说，batch 中有 `N` 段文本，其中只有一个是匹配文本。模型需要把正确文本的相似度推高，把其他文本相似度压低。

这和普通分类很像：

```text
输入：image_i
类别：text_0, text_1, ..., text_{N-1}
正确类别：text_i
```

区别是类别不是固定标签，而是当前 batch 里的文本 embedding。

优点：

1. 不需要人工类别标签。
2. batch 越大，负样本越多。
3. 学到的是开放语义空间。

问题：

1. batch 内可能有 false negative。
2. 对数据质量很敏感。
3. 大 batch 训练成本高。

## 2.9 Zero-shot 图像分类

CLIP 的一个重要能力是 zero-shot classification。

传统分类模型最后一层是固定类别分类头。CLIP 不需要训练新的分类头，而是把类别名称写成文本 prompt。

例如类别：

```text
dog, cat, car
```

构造 prompt：

```text
a photo of a dog
a photo of a cat
a photo of a car
```

流程：

1. 用 text encoder 编码所有类别 prompt。
2. 用 image encoder 编码待分类图片。
3. 计算图片向量和每个类别文本向量的相似度。
4. 选择相似度最高的类别。

示意代码：

```python
image_features = encode_image(image)
text_features = encode_text(class_prompts)

image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

scores = image_features @ text_features.T
pred = scores.argmax(dim=-1)
```

这就是开放词表分类的基础。

## 2.10 Prompt engineering 对 CLIP 的影响

CLIP 的 zero-shot 效果会受 prompt 影响。

例如：

```text
dog
a photo of a dog
a blurry photo of a dog
a photo of a small dog
```

不同 prompt 可能产生不同文本 embedding，从而影响分类结果。

常见做法是 prompt ensemble：对同一个类别构造多个模板，然后平均文本向量。

例如：

```text
a photo of a {label}
a blurry photo of a {label}
a close-up photo of a {label}
```

这样可以提升鲁棒性。

## 2.11 图文检索

CLIP 很适合图文检索。

### 2.11.1 Text-to-image retrieval

输入文本 query，检索最匹配图片。

流程：

1. 离线编码图片库，得到 image embeddings。
2. 在线编码文本 query。
3. 计算 query embedding 和图片 embedding 的相似度。
4. 返回 Top-K 图片。

### 2.11.2 Image-to-text retrieval

输入图片，检索最匹配文本描述。

流程类似，只是 query 变成图片。

### 2.11.3 评估指标

常见指标：

1. Recall@K。
2. Mean Reciprocal Rank。
3. Top-1 Accuracy。
4. Median Rank。

面试中如果讲项目，建议至少说清楚 Recall@K 和 bad case 分析。

## 2.12 CLIP 和 VLM 的关系

CLIP 本身不是对话式 VLM。它主要输出图像和文本的 embedding，相似度用于检索、分类和对齐。

VLM 通常需要进一步把视觉特征接入 LLM：

```text
vision encoder -> projector -> LLM
```

CLIP 的 vision encoder 可以作为 VLM 的视觉编码器，提供已经具备语义对齐能力的视觉特征。

区别：

1. CLIP 擅长对齐和检索。
2. VLM 擅长图文对话、问答和复杂指令遵循。
3. CLIP 双塔交互较浅，VLM 通常通过 LLM 做更强语言推理。

## 2.13 一个最小 CLIP 训练骨架

下面是教学版结构，不追求完整工程性能。

```python
import torch
from torch import nn
import torch.nn.functional as F


class MiniCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_dim, text_dim, embed_dim):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, images):
        x = self.image_encoder(images)
        x = self.image_proj(x)
        return F.normalize(x, dim=-1)

    def encode_text(self, input_ids, attention_mask=None):
        x = self.text_encoder(input_ids, attention_mask=attention_mask)
        x = self.text_proj(x)
        return F.normalize(x, dim=-1)

    def forward(self, images, input_ids, attention_mask=None):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        return {"loss": loss, "logits": logits}
```

这个骨架包含：

1. image encoder。
2. text encoder。
3. projection 到共享空间。
4. 向量归一化。
5. 相似度矩阵。
6. 双向 contrastive loss。

## 2.14 最小可运行 CLIP loss / 检索 demo

下面的 demo 不依赖 PyTorch，用标准库手写 L2 normalize、相似度矩阵、双向 cross entropy、Recall@1、MRR 和 zero-shot prompt ensemble。它不是训练真实图像模型，而是把 CLIP loss 的数学结构跑通。

```python
from math import exp, log, sqrt


def normalize(vector):
    norm = sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def dot(left, right):
    return sum(x * y for x, y in zip(left, right))


def softmax(scores):
    max_score = max(scores)
    values = [exp(score - max_score) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def cross_entropy_row(scores, target):
    return -log(softmax(scores)[target])


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def rank_of_target(scores, target):
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return order.index(target) + 1


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

similarity = [[dot(image, text) for text in text_features] for image in image_features]
logits = [[score * logit_scale for score in row] for row in similarity]
labels = list(range(len(logits)))

loss_i2t = sum(
    cross_entropy_row(row, target) for row, target in zip(logits, labels)
) / len(labels)
loss_t2i = sum(
    cross_entropy_row(row, target) for row, target in zip(transpose(logits), labels)
) / len(labels)
clip_loss = (loss_i2t + loss_t2i) / 2

ranks_i2t = [rank_of_target(row, target) for row, target in zip(logits, labels)]
ranks_t2i = [
    rank_of_target(row, target) for row, target in zip(transpose(logits), labels)
]
recall_at_1 = sum(rank == 1 for rank in ranks_i2t) / len(ranks_i2t)
mrr = sum(1 / rank for rank in ranks_i2t) / len(ranks_i2t)

zero_shot_prompts = {
    "dog": [[0.86, 0.06, 0.01, 0.00], [0.82, 0.10, 0.02, 0.03]],
    "receipt": [[0.04, 0.90, 0.12, 0.01], [0.08, 0.84, 0.10, 0.04]],
    "chart": [[0.02, 0.10, 0.90, 0.12], [0.01, 0.12, 0.86, 0.14]],
}

class_vectors = {}
for class_name, variants in zero_shot_prompts.items():
    normalized_variants = [normalize(vector) for vector in variants]
    averaged = [
        sum(values) / len(values) for values in zip(*normalized_variants)
    ]
    class_vectors[class_name] = normalize(averaged)

query_image = normalize([0.03, 0.12, 0.90, 0.10])
class_scores = {
    class_name: dot(query_image, vector)
    for class_name, vector in class_vectors.items()
}
pred_label = max(class_scores, key=class_scores.get)

similarity_matrix = [[round(value, 3) for value in row] for row in similarity]
metrics = {
    "loss_i2t": round(loss_i2t, 6),
    "loss_t2i": round(loss_t2i, 6),
    "clip_loss": round(clip_loss, 6),
    "recall_at_1": round(recall_at_1, 3),
    "mrr": round(mrr, 3),
}
checks = {
    "diagonal_is_best": all(rank == 1 for rank in ranks_i2t + ranks_t2i),
    "loss_small": clip_loss < 0.001,
    "retrieval_ok": recall_at_1 == 1.0 and mrr == 1.0,
    "zero_shot_ok": pred_label == "chart",
}

print(f"similarity_matrix={similarity_matrix}")
print(f"metrics={metrics}")
print(f"ranks_i2t={dict(zip(image_names, ranks_i2t))}")
print(f"ranks_t2i={dict(zip(text_names, ranks_t2i))}")
print(
    "zero_shot_scores="
    + str({key: round(value, 3) for key, value in class_scores.items()})
)
print(f"pred_label={pred_label}")
print(f"checks={checks}")
print(f"gate_pass={all(checks.values())}")
```

期望输出中，相似度矩阵的对角线接近 1，`loss_i2t` 和 `loss_t2i` 都很小，image-to-text 和 text-to-image 的正确配对 rank 都是 1，zero-shot 查询会预测为 `chart`，最终输出 `gate_pass=True`。

这个 demo 对应真实项目里的三层检查：

1. loss 检查：正样本是否比 batch 内负样本明显更相似。
2. retrieval 检查：正确图文对能否排到 Top-1 或 Top-K。
3. zero-shot 检查：类别 prompt 是否真的能作为开放词表分类头。

## 2.15 数据质量的重要性

CLIP 依赖大规模图文对，但图文对噪声很大。

常见问题：

1. 文本描述和图片不匹配。
2. 文本太短，只是文件名或无意义标签。
3. 图片含商业水印、拼接图。
4. 多语言混杂。
5. 重复样本多。
6. 有害或偏见内容。

数据质量会直接影响对齐质量。模型可能学到 spurious correlation，例如水印、背景、拍摄风格，而不是真正语义。

## 2.16 CLIP 的局限

CLIP 很强，但不是万能视觉理解模型。

局限包括：

1. 细粒度计数能力弱。
2. OCR 能力有限。
3. 空间关系理解不稳定。
4. 对组合关系敏感，例如“红色球在蓝色盒子上”。
5. 容易受 prompt 影响。
6. 对训练数据偏差敏感。
7. 只能输出相似度，不擅长长回答。

例如：

```text
a dog chasing a cat
a cat chasing a dog
```

这两句话词汇相似，但关系相反。CLIP 可能不如专门的 VLM 或 grounding 模型稳定。

## 2.17 CLIP 项目怎么讲

如果做一个 CLIP 图文检索项目，简历表达可以是：

```text
基于 CLIP 构建图文检索系统，使用 image/text encoder 生成归一化 embedding，通过 cosine similarity 和 FAISS 实现 text-to-image 与 image-to-text 检索，使用 Recall@K、MRR 和 Top-1 Accuracy 评估，并分析中文 prompt、细粒度识别、OCR 和空间关系场景的 bad cases。
```

面试时建议讲清楚：

1. 数据集是什么。
2. embedding 如何生成。
3. 检索索引如何构建。
4. 指标如何计算。
5. 哪些 bad case 最常见。
6. 如何改进。

## 2.18 常见面试题

### 问题一：CLIP 的核心思想是什么？

回答模板：

```text
CLIP 的核心是图文对比学习。它用 image encoder 和 text encoder 分别编码图片和文本，把它们投影到同一个 embedding 空间，让匹配图文对相似度高，不匹配图文对相似度低。训练好后可以做图文检索和 zero-shot 图像分类。
```

### 问题二：CLIP 的 loss 怎么算？

回答模板：

```text
一个 batch 有 N 对图文，分别得到 N 个 image embeddings 和 N 个 text embeddings，归一化后计算 N x N 相似度矩阵。对角线是正样本，非对角线是 batch 内负样本。loss 通常是双向 cross entropy：image-to-text 和 text-to-image 两个方向取平均。
```

### 问题三：CLIP 如何做 zero-shot 分类？

回答模板：

```text
先把类别名称写成 prompt，比如 a photo of a dog，用 text encoder 编码每个类别 prompt；再用 image encoder 编码待分类图片；最后计算图片向量和各类别文本向量的相似度，选择相似度最高的类别。
```

### 问题四：CLIP 为什么适合检索？

回答模板：

```text
因为 CLIP 是双塔结构，图像和文本可以分别编码成同一个语义空间里的向量。图片库可以离线编码并建索引，查询时只需要编码 query 并做向量相似度搜索，所以很适合 text-to-image 和 image-to-text retrieval。
```

### 问题五：CLIP 有哪些局限？

回答模板：

```text
CLIP 擅长全局语义对齐，但细粒度计数、OCR、空间关系、复杂组合关系和长文本推理较弱。它主要输出相似度，不是对话模型，也容易受 prompt 和训练数据偏差影响。
```

## 2.19 小练习

1. 写出 CLIP 双塔结构的数据流。
2. 给定 `image_features [N,d]` 和 `text_features [N,d]`，手写相似度矩阵。
3. 手写 CLIP 双向 cross entropy loss。
4. 用 3 个类别 prompt 模拟 zero-shot 分类。
5. 设计一个 text-to-image retrieval 的 Recall@K 评估。
6. 列出 5 类 CLIP bad case。
7. 说明 CLIP 和 VLM 的区别。
8. 修改本章 demo 的 temperature，观察 `clip_loss` 和 softmax 置信度如何变化。

## 2.20 本章总结

CLIP 是理解多模态对齐的基础。它通过 image encoder 和 text encoder 构建双塔模型，用对比学习把图像和文本投影到同一个语义空间。相似度矩阵的对角线是正样本，非对角线是 batch 内负样本，双向 cross entropy 让匹配图文靠近、不匹配图文远离。

需要记住的主线：

1. CLIP 解决的是图文语义对齐问题。
2. 双塔结构支持高效检索和离线索引。
3. 对比学习使用 in-batch negatives。
4. zero-shot 分类通过类别 prompt 和相似度完成。
5. CLIP 强在开放语义和检索，弱在细粒度推理、OCR、空间关系和长回答。

下一章会进入 vision encoder，系统讲 CNN、ViT、patch embedding、视觉特征层级、CLIP vision tower 和 VLM 中视觉编码器的选择。
