# 第三章：Vision Encoder

Vision encoder 是多模态模型感知图像的入口。无论是 CLIP 图文检索，还是 LLaVA、Qwen-VL、InternVL 这类视觉语言模型，第一步通常都是把原始图片变成一组视觉特征。后续 projector、LLM、cross-attention 或生成模块处理的不是图片像素本身，而是 vision encoder 输出的 visual tokens 或 image embedding。

本章目标是讲清 vision encoder 的作用、CNN 和 ViT 的区别、patch embedding 的直觉、CLIP/SigLIP 这类对齐型视觉编码器、视觉特征层选择、分辨率与 token 数、OCR 和细粒度场景的挑战，以及在 VLM 中如何选择和使用 vision encoder。

## 0. 本讲资料边界与第二轮精修口径

第二轮精修前，本讲按 `WRITING_PLAN.md` 核对了 ResNet、ViT、CLIP 官方代码和 SigLIP 论文资料。这里不做完整视觉模型史，而是围绕大模型算法岗最常追问的 vision encoder 入口来写：

1. ResNet 代表 CNN 残差学习路线，重点是 residual connection 让更深网络更容易优化，适合解释 CNN 的局部归纳偏置和分层视觉特征。
2. ViT 论文的关键口径是：把图片切成 patch 序列，直接送入 Transformer；当有大规模预训练数据时，纯 Transformer 视觉模型可以取得很强效果。
3. OpenAI CLIP 的 ViT vision tower 使用 `Conv2d(kernel_size=patch_size, stride=patch_size)` 做 patch embedding，加 class embedding 和 positional embedding，再过 Transformer，并取 class token 投影为全局 image embedding。
4. SigLIP 仍属于图文对齐视觉编码器路线，但用 pairwise sigmoid loss 替代全局 softmax 归一化，工程上常作为现代 VLM 的视觉塔选择之一。
5. 本章聚焦 CNN / ViT / patch embedding / CLS 与 patch tokens / 分辨率 token 成本 / CLIP vision tower / projector 前的视觉特征选择，不展开检测分割网络、DETR、SAM、OCR 专用模型、视频 encoder 或 VLM 全架构细节。

本章第二轮重点是把 vision encoder 相关 shape、token 数、参数量、attention 成本和分辨率取舍写成可检查公式，并补一个无依赖 demo，帮助读者在面试中手算视觉 token 和成本。

## 3.1 Vision Encoder 解决什么问题

图片原始输入通常是像素矩阵：

```text
image: [H, W, 3]
```

或者在 PyTorch 中：

```text
image_tensor: [B, 3, H, W]
```

LLM 不能直接理解像素矩阵。vision encoder 的作用是把图片转换成更抽象的视觉表示。

常见输出形式：

1. 全局 image embedding：用于检索或分类。
2. patch-level visual tokens：用于 VLM 问答和细粒度理解。
3. 多层视觉特征：用于检测、分割或高分辨率任务。

面试回答：

```text
Vision encoder 的作用是把原始图像像素编码成模型可处理的视觉特征。CLIP 这类模型通常需要全局 image embedding 做图文对齐和检索，VLM 则更常使用 patch-level visual tokens，再通过 projector 接入 LLM，让语言模型基于图像内容回答问题。
```

## 3.2 CNN 路线

早期视觉模型主要基于 CNN。CNN 使用卷积核在图片上滑动，提取局部纹理、边缘、形状和更高层语义。

典型 CNN 特点：

1. 局部感受野。
2. 权重共享。
3. 平移等变性。
4. 分层特征。
5. 对图像结构有强 inductive bias。

常见模型：

1. AlexNet。
2. VGG。
3. ResNet。
4. EfficientNet。

CNN 的优点：

1. 对图像局部结构建模自然。
2. 数据效率相对较好。
3. 在检测、分割等任务中有长期积累。

缺点：

1. 全局关系建模不如 Transformer 直接。
2. 和文本 Transformer 结构差异较大。
3. 在大规模图文预训练时代，ViT 路线更常见。

CLIP 早期既用过 ResNet，也用过 ViT。现代 VLM 更多使用 ViT 或 ViT 变体作为视觉塔。

## 3.3 Vision Transformer 的基本思想

ViT 的核心思想是把图片切成 patch，把 patch 当成视觉 token，再用 Transformer 处理。

流程：

```text
image -> patches -> patch embeddings -> Transformer encoder -> visual tokens
```

例如图片大小是 `224 x 224`，patch size 是 `16 x 16`，则 patch 数量是：

```text
(224 / 16) * (224 / 16) = 14 * 14 = 196
```

如果加一个 CLS token，总 token 数是 197。

ViT 的优点：

1. 和 LLM 都是 Transformer，结构统一。
2. 更容易扩展到大规模预训练。
3. self-attention 能建模全局关系。
4. 输出 patch tokens 适合接入 VLM。

缺点：

1. token 数随分辨率平方增长。
2. 对数据规模和训练策略敏感。
3. 对小目标、OCR、细粒度细节需要额外设计。

## 3.4 Patch Embedding

Patch embedding 的作用是把每个图像 patch 变成向量。

直觉上，一个 patch 是一个小图片块：

```text
patch: [P, P, 3]
```

把它展平后通过线性层：

```text
patch vector -> linear -> patch embedding
```

PyTorch 中常用一个卷积层实现 patch embedding：

```python
from torch import nn


patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=hidden_size,
    kernel_size=patch_size,
    stride=patch_size,
)
```

如果输入是：

```text
x: [B, 3, H, W]
```

输出是：

```text
x: [B, hidden_size, H/P, W/P]
```

再展平为 token 序列：

```python
x = patch_embed(images)          # [B, d, H/P, W/P]
x = x.flatten(2).transpose(1, 2) # [B, N, d]
```

其中 `N = (H/P) * (W/P)`。

### 3.4.1 关键公式与 Vision Encoder shape 速查

设输入图像 batch 为 `X: [B, C, H, W]`，patch size 为 `P`。如果先 resize / padding 到 patch 网格，patch 网格大小可以近似写成：

```math
N_h=\left\lceil\frac{H}{P}\right\rceil,\qquad
N_w=\left\lceil\frac{W}{P}\right\rceil
```

视觉 patch token 数是：

```math
N_{\mathrm{patch}}=N_h N_w
```

如果不做 padding，而是使用 kernel size 和 stride 都等于 `P` 的卷积，输出网格更接近：

```math
H_o=\left\lfloor\frac{H-P}{P}\right\rfloor+1,\qquad
W_o=\left\lfloor\frac{W-P}{P}\right\rfloor+1
```

这就是为什么工程实现里要明确 resize、crop、padding 和 patch size，否则视觉 token 数会和预期不一致。

一个 patch 展平后的维度是：

```math
d_{\mathrm{patch}}=C P^2
```

线性 patch embedding 可以写成：

```math
z_i=p_i W_e+b_e,\qquad
W_e\in\mathbb{R}^{d_{\mathrm{patch}}\times d_v}
```

其中 `d_v` 是 vision hidden size。用 `Conv2d(C, d_v, kernel_size=P, stride=P)` 实现时，权重 shape 等价于 `[d_v, C, P, P]`。

ViT 加 CLS token 和位置 embedding 后，输入 Transformer 的序列可以写成：

```math
X_0=[x_{\mathrm{cls}};z_1;\ldots;z_N]+E_{\mathrm{pos}},\qquad
E_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times d_v}
```

如果不用 CLS，而是把 patch tokens 做平均池化或 attention pooling，则全局 image embedding 可以抽象为：

```math
g=\mathrm{pool}(Z_1,\ldots,Z_N)
```

VLM 通常不只需要全局 embedding，而是保留 patch tokens，再通过 projector 映射到 LLM hidden size：

```math
Y=Z W_p+b_p,\qquad
Y\in\mathbb{R}^{B\times N\times d_l}
```

其中 `Z: [B,N,d_v]`，`d_l` 是 LLM hidden size。

视觉 token 进入 Transformer 后，self-attention 的主要 score cell 数近似为：

```math
C_{\mathrm{vis}}\propto (N_{\mathrm{patch}}+1)^2
```

所以当分辨率从 `H x W` 放大到 `rH x rW`，patch size 不变时，token 数大约变成 `r^2` 倍，而 attention score cell 数大约变成 `r^4` 倍。面试里解释高分辨率 VLM 成本时，这个量级判断很关键。

CNN 残差块的核心公式可以写成：

```math
h_{l+1}=h_l+F_l(h_l)
```

它体现了 ResNet 的残差学习思想：层不必从零学习完整映射，而是学习相对输入的增量。对比 ViT 时，可以说 CNN 有局部卷积归纳偏置，ViT 更依赖 patch token 和 attention 做全局建模。

## 3.5 CLS Token 和 Patch Tokens

ViT 常见输出包括：

1. CLS token。
2. Patch tokens。

CLS token 通常代表整张图的全局语义，适合分类或图文对齐。

Patch tokens 保留空间结构，适合 VLM、定位、OCR、细粒度问答。

CLIP 检索常使用全局 image embedding；VLM 更常把 patch tokens 经过 projector 后送入 LLM。

例如：

```text
visual_tokens: [B, N, d_v]
projector(visual_tokens): [B, N, d_llm]
```

## 3.6 图像分辨率与视觉 token 数

视觉 token 数量对 VLM 成本影响很大。

如果 patch size 固定为 14：

```text
image 224 x 224 -> 16 x 16 = 256 tokens
image 448 x 448 -> 32 x 32 = 1024 tokens
image 896 x 896 -> 64 x 64 = 4096 tokens
```

分辨率越高，细节越多，但 token 数也快速增加。送入 LLM 后，视觉 token 会占用上下文窗口和 attention 计算。

工程 trade-off：

1. 高分辨率提升 OCR、小目标和图表理解。
2. 高分辨率增加计算、显存和延迟。
3. 低分辨率更快，但容易丢失细节。

这也是为什么很多 VLM 会做动态分辨率、图片切块、token pruning 或视觉 token 压缩。

## 3.7 CLIP Vision Tower

很多 VLM 会直接使用 CLIP 的 vision encoder 作为视觉塔。

原因：

1. CLIP 已经通过图文对齐学到强语义视觉特征。
2. 视觉特征和语言语义更接近。
3. 开源生态成熟。
4. 下游 VLM 训练更容易起步。

结构：

```text
image -> CLIP vision tower -> visual tokens -> projector -> LLM
```

需要注意：CLIP 原本训练目标是图文对齐，不是专门为 OCR、检测或复杂视觉推理设计。因此在某些细粒度任务上可能需要更强的 vision encoder、更高分辨率或额外训练数据。

## 3.8 SigLIP 的基本直觉

SigLIP 可以理解为 CLIP 类图文对齐模型的一条重要改进路线。CLIP 通常使用 softmax 对比学习，在 batch 内构造 `N x N` 相似度矩阵；SigLIP 使用 sigmoid loss，把图文匹配看成多个二分类问题。

直觉区别：

1. CLIP：一张图在 batch 中选择正确文本。
2. SigLIP：每个图文对判断匹配或不匹配。

SigLIP 的优势之一是对 batch size 和分布式训练形态可能更友好，因此很多现代 VLM 会使用 SigLIP vision tower。

面试中不需要背具体实现细节，但要知道：SigLIP 仍然是图文对齐视觉编码器路线，常作为 VLM 的视觉塔选择之一。

## 3.9 视觉特征层选择

Vision encoder 有多层输出。不同层的语义不同：

1. 浅层更偏边缘、纹理、局部细节。
2. 中层包含形状和部件。
3. 深层更偏全局语义。

VLM 中经常需要选择某一层或多层视觉特征。

选择深层特征：

1. 语义强。
2. 适合图像描述和整体理解。
3. 可能丢失细节。

选择中间层或多层特征：

1. 保留更多局部信息。
2. 有利于 OCR、定位、图表和细粒度任务。
3. 接入和训练更复杂。

工程上通常通过验证集和 bad case 来决定使用哪层视觉特征，而不是只凭直觉。

## 3.10 Projector：把视觉维度接到 LLM

Vision encoder 输出维度通常和 LLM hidden size 不同，需要 projector。

最简单 projector 是线性层：

```python
projector = nn.Linear(vision_hidden_size, llm_hidden_size)
```

也可以用 MLP：

```python
projector = nn.Sequential(
    nn.Linear(vision_hidden_size, llm_hidden_size),
    nn.GELU(),
    nn.Linear(llm_hidden_size, llm_hidden_size),
)
```

作用：

1. 维度对齐。
2. 把视觉特征适配到 LLM 表示空间。
3. 在冻结 vision encoder 和 LLM 时，projector 是最小可训练桥接模块。

很多早期 VLM 训练会先冻结 vision encoder 和 LLM，只训练 projector，让 visual tokens 初步接入语言模型。

## 3.11 图片预处理

Vision encoder 对输入预处理很敏感。

常见步骤：

1. resize。
2. center crop 或 padding。
3. normalize。
4. 转成 `[B, 3, H, W]`。

不同 vision encoder 的 normalize 均值和方差可能不同。如果预处理和预训练时不一致，效果会明显下降。

例如：

```python
pixel_values = processor(images, return_tensors="pt")["pixel_values"]
```

使用开源模型时，优先使用对应 processor，不要手写不确定的预处理流程。

## 3.12 OCR 和细粒度场景为什么难

OCR、图表、截图、小目标和空间关系是 VLM 的难点。

原因包括：

1. 输入分辨率不足，文字被压缩。
2. patch size 太大，小字落在少数 patch 中。
3. vision encoder 预训练目标不强调 OCR。
4. 视觉 token 被压缩，细节丢失。
5. LLM 语言先验会补全不存在的内容。

改进方向：

1. 提高分辨率。
2. 图片切块。
3. 使用 OCR 专门数据训练。
4. 结合外部 OCR 工具。
5. 引入 grounding 和引用证据。

## 3.13 Vision Encoder 选择标准

选择 vision encoder 时要考虑：

1. 是否经过图文对齐预训练。
2. 支持的分辨率。
3. 输出 token 数。
4. 视觉 hidden size。
5. OCR 和细粒度能力。
6. 推理速度和显存。
7. 开源权重和生态支持。
8. 与 LLM/projector 的兼容性。

如果目标是图文检索，CLIP/SigLIP 对齐特征很重要。如果目标是文档 OCR 或图表理解，只靠普通 CLIP vision tower 可能不够。

## 3.14 一个最小 ViT Patch Embedding 示例

```python
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, hidden_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # [B, d, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, N, d]
        return x


images = torch.randn(2, 3, 224, 224)
patch_embed = PatchEmbedding()
tokens = patch_embed(images)
print(tokens.shape)  # [2, 196, 768]
```

这个例子说明：ViT 的第一步就是把图像变成 token 序列。

## 3.15 最小可运行 Vision Encoder shape / patch demo

下面的 demo 不依赖 PyTorch，用标准库手算 patch grid、patch embedding 参数量、CLS / positional embedding shape、视觉 attention 成本、projector shape 和 OCR / 图表任务的分辨率取舍。

```python
from math import ceil


def patch_grid(height, width, patch):
    return ceil(height / patch), ceil(width / patch)


def patch_tokens(height, width, patch):
    grid_h, grid_w = patch_grid(height, width, patch)
    return grid_h * grid_w


def patch_vector_dim(channels, patch):
    return channels * patch * patch


def patch_embedding_params(channels, patch, hidden, bias=True):
    params = patch_vector_dim(channels, patch) * hidden
    if bias:
        params += hidden
    return params


def position_embedding_shape(height, width, patch, hidden, cls=True):
    tokens = patch_tokens(height, width, patch) + (1 if cls else 0)
    return (tokens, hidden)


def attention_cells(tokens):
    return tokens * tokens


def projector_params(vision_hidden, llm_hidden, bias=True):
    params = vision_hidden * llm_hidden
    if bias:
        params += llm_hidden
    return params


resolutions = [(224, 224), (336, 336), (448, 448), (672, 448)]
patch = 14
hidden = 1024
resolution_table = []
for height, width in resolutions:
    grid_h, grid_w = patch_grid(height, width, patch)
    num_tokens = grid_h * grid_w
    with_cls = num_tokens + 1
    resolution_table.append(
        {
            "resolution": f"{height}x{width}",
            "grid": (grid_h, grid_w),
            "patch_tokens": num_tokens,
            "with_cls": with_cls,
            "attn_cells": attention_cells(with_cls),
        }
    )

base_cells = resolution_table[0]["attn_cells"]
for row in resolution_table:
    row["attn_vs_224"] = round(row["attn_cells"] / base_cells, 2)

shape_audit = {
    "image": (2, 3, 336, 336),
    "patch_grid": patch_grid(336, 336, patch),
    "patch_vector_dim": patch_vector_dim(3, patch),
    "patch_embedding_weight": (hidden, 3, patch, patch),
    "patch_tokens": patch_tokens(336, 336, patch),
    "sequence_with_cls": (2, patch_tokens(336, 336, patch) + 1, hidden),
    "position_embedding": position_embedding_shape(336, 336, patch, hidden, cls=True),
    "projected_tokens": (2, patch_tokens(336, 336, patch), 4096),
}

costs = {
    "patch_embed_params_bias_false": patch_embedding_params(3, patch, hidden, bias=False),
    "patch_embed_params_bias_true": patch_embedding_params(3, patch, hidden, bias=True),
    "projector_params": projector_params(hidden, 4096, bias=True),
}

task_tradeoffs = [
    {"name": "image_caption", "detail": 0.35, "tokens": resolution_table[0]["patch_tokens"]},
    {"name": "chart_qa", "detail": 0.75, "tokens": resolution_table[1]["patch_tokens"]},
    {"name": "dense_ocr", "detail": 0.92, "tokens": resolution_table[2]["patch_tokens"]},
]
for item in task_tradeoffs:
    item["needs_high_res"] = item["detail"] >= 0.7
    item["over_1k_tokens"] = item["tokens"] > 1000

checks = {
    "336_grid_ok": shape_audit["patch_grid"] == (24, 24),
    "cls_position_ok": shape_audit["position_embedding"] == (577, 1024),
    "attn_cost_grows": resolution_table[2]["attn_vs_224"] > resolution_table[1]["attn_vs_224"] > 1.0,
    "projector_shape_ok": shape_audit["projected_tokens"] == (2, 576, 4096),
    "ocr_flagged_high_res": task_tradeoffs[2]["needs_high_res"] and task_tradeoffs[2]["over_1k_tokens"],
}

print(f"resolution_table={resolution_table}")
print(f"shape_audit={shape_audit}")
print(f"costs={costs}")
print(f"task_tradeoffs={task_tradeoffs}")
print(f"checks={checks}")
print(f"gate_pass={all(checks.values())}")
```

期望输出中，`336x336`、patch size 为 `14` 会得到 `24 x 24 = 576` 个 patch token；加 CLS 后 positional embedding shape 是 `(577, 1024)`；`448x448` 的视觉 attention score cell 数约为 `224x224` 的 `15.91` 倍；projector 会把 `(2,576,1024)` 视觉 token 映射成 `(2,576,4096)`。

这个 demo 对应三个面试动作：

1. 先手算 patch token，不要只说“高分辨率更贵”。
2. 分清全局 CLS embedding 和 patch tokens 的用途。
3. 把任务需求和成本关联起来：caption 可以低分辨率，图表和 OCR 往往要更高分辨率，但会显著增加 token 和 attention 成本。

## 3.16 常见面试题

### 问题一：Vision encoder 在 VLM 中的作用是什么？

回答模板：

```text
Vision encoder 负责把原始图像像素编码成视觉特征。VLM 通常使用它输出的 patch-level visual tokens，再通过 projector 映射到 LLM hidden size，让语言模型能基于图像信息进行问答和推理。
```

### 问题二：CNN 和 ViT 的主要区别是什么？

回答模板：

```text
CNN 通过卷积提取局部特征，有局部感受野和权重共享，对图像结构有强 inductive bias；ViT 把图片切成 patch token，用 Transformer self-attention 建模全局关系，更容易和语言 Transformer 架构统一，也更适合大规模预训练和 VLM 接入。
```

### 问题三：Patch embedding 是什么？

回答模板：

```text
Patch embedding 是把图片切成固定大小 patch，并把每个 patch 映射成一个向量。实现上常用 kernel size 和 stride 都等于 patch size 的 Conv2d，把 [B,3,H,W] 转成 [B,N,d] 的视觉 token 序列。
```

### 问题四：为什么高分辨率会增加 VLM 成本？

回答模板：

```text
因为视觉 token 数大约随图像高宽的乘积增长。patch size 固定时，分辨率翻倍，token 数大约变成 4 倍。视觉 token 进入 LLM 后会占用上下文长度，并增加 attention 计算和显存。
```

### 问题五：为什么 OCR 对 VLM 很难？

回答模板：

```text
OCR 依赖细粒度文字信息，而普通 vision encoder 可能在 resize、patch embedding 和 token 压缩过程中丢失小字细节。CLIP 这类模型主要学全局图文语义，不一定专门优化 OCR。改进通常需要高分辨率、切图、OCR 数据或外部 OCR 工具。
```

## 3.17 小练习

1. 给定图片大小 `224 x 224`、patch size `16`，计算 patch token 数。
2. 写一个 `PatchEmbedding` 模块，把 `[B,3,H,W]` 转成 `[B,N,d]`。
3. 比较 CLS token 和 patch tokens 的用途。
4. 解释为什么 VLM 常用 CLIP 或 SigLIP vision tower。
5. 设计一个 OCR bad case，用来测试 vision encoder 是否保留细节。
6. 分析分辨率从 224 提升到 448 后，视觉 token 数和推理成本如何变化。
7. 修改本章 demo 的 patch size，比较 `P=14` 和 `P=16` 的 token 数、position embedding shape 和 attention cell 数。

## 3.18 本章总结

Vision encoder 是多模态系统的视觉入口。它把图片像素转换成全局 embedding 或 patch-level visual tokens。CLIP/SigLIP 这类图文对齐模型提供了强视觉语义基础，ViT 通过 patch embedding 把图片变成 token 序列，便于接入 LLM。

需要记住：

1. CNN 强在局部结构和视觉 inductive bias。
2. ViT 把图片变成 patch token 序列，适合和 Transformer LLM 结合。
3. 分辨率越高，视觉 token 数和成本越高。
4. VLM 通常需要 projector 把视觉特征映射到 LLM hidden size。
5. OCR、图表、小目标和空间关系需要更高分辨率、更细粒度特征或专门数据。
6. 选择 vision encoder 要综合对齐能力、分辨率、token 数、速度、显存和任务需求。

下一章会进入 VLM 架构，讲清 visual tokens 如何接入 LLM、projector、cross-attention、image token、chat template 和多轮图文对话结构。
