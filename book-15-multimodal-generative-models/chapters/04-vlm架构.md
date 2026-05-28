# 第四章：VLM 架构

VLM，也就是 Vision-Language Model，是把视觉能力接入语言模型的一类模型。它既要看懂图片，又要用自然语言回答问题、描述内容、执行指令和进行推理。和 CLIP 不同，CLIP 主要输出图文相似度；VLM 通常要生成文本回答，因此需要把视觉表示接入 LLM 的生成过程。

本章重点讲 VLM 的主流架构：vision encoder + connector + LLM、LLaVA 风格的 visual token 拼接、Flamingo 的 cross-attention、BLIP-2 的 Q-Former、Perceiver Resampler、image token 的处理、多图和多轮对话结构，以及工程上的分辨率、token 数、冻结策略和训练阶段设计。

## 4.1 VLM 解决什么问题

CLIP 能判断图文是否匹配，但不能自然回答复杂问题。

例如：

```text
这张图里有几个人？他们在做什么？左边的人手里拿着什么？
```

这类任务需要：

1. 识别图像内容。
2. 理解用户问题。
3. 结合视觉和语言推理。
4. 生成自然语言答案。

VLM 的目标就是把视觉信息变成 LLM 能使用的上下文。

面试回答：

```text
VLM 的目标是让语言模型能够使用视觉信息进行问答、描述、推理和指令遵循。常见做法是用 vision encoder 提取视觉特征，再通过 projector、Q-Former 或 cross-attention 等 connector 接入 LLM，使 LLM 在生成文本时能条件化于图片内容。
```

## 4.2 VLM 的基本三段式结构

最常见 VLM 可以抽象为三段：

```text
image -> vision encoder -> connector -> LLM -> text answer
```

其中：

1. Vision encoder：把图片转成 visual tokens。
2. Connector：把视觉特征转换成 LLM 可用的表示。
3. LLM：结合文本 prompt 和视觉 tokens 生成回答。

更具体：

```text
image -> CLIP/SigLIP ViT -> visual tokens -> projector -> LLM hidden states
text  -> tokenizer -> text tokens -> LLM hidden states
```

关键问题：视觉 tokens 如何和文本 tokens 结合。

常见路线：

1. 直接拼接 visual tokens 和 text tokens。
2. 用 cross-attention 让 LLM 访问视觉特征。
3. 用 Q-Former 或 resampler 压缩视觉 tokens。
4. 把图像离散化成特殊 tokens。

## 4.3 LLaVA 风格：Projector + Token 拼接

LLaVA 代表了一类简单有效的 VLM 架构。

结构：

```text
image -> vision encoder -> visual tokens -> projector -> LLM token space
prompt -> text tokens -> LLM
```

然后把视觉 tokens 插入到文本序列中。

例如用户输入：

```text
<image>
请描述这张图片。
```

内部可能变成：

```text
[visual_token_1, visual_token_2, ..., visual_token_N, text_token_1, ...]
```

Projector 通常是线性层或 MLP：

```python
projector = nn.Sequential(
    nn.Linear(vision_hidden_size, llm_hidden_size),
    nn.GELU(),
    nn.Linear(llm_hidden_size, llm_hidden_size),
)
```

优点：

1. 结构简单。
2. 训练和实现容易。
3. 可以复用现成 vision encoder 和 LLM。
4. 适合 instruction tuning。

缺点：

1. 视觉 token 多时占用大量上下文。
2. 长图、多图、高分辨率成本高。
3. 视觉和语言融合完全依赖 LLM self-attention。

## 4.4 Image Token 和占位符

VLM 数据中常出现 `<image>` 占位符。

它有两层含义：

1. 文本模板中的特殊标记，告诉模型这里有图片。
2. 模型内部用于插入视觉 tokens 的位置。

例如：

```text
User: <image> 这张图里有什么？
Assistant: 图中有一只猫坐在沙发上。
```

训练时要保证：

1. 图片数量和 `<image>` 数量一致。
2. visual tokens 插入位置正确。
3. label mask 只训练 assistant 回答。
4. 多轮对话中图片引用不混乱。

常见 bug：模板里有 `<image>`，但 batch 里没有对应图片；或者图片 tokens 插入后 labels 没有同步对齐。

## 4.5 Flamingo：Cross-Attention 路线

Flamingo 代表另一类思路：不一定把所有视觉 tokens 直接拼进 LLM 主序列，而是在 LLM 层中插入 cross-attention，让语言 token 可以 attend 到视觉特征。

结构直觉：

```text
text hidden states -> self-attention
text hidden states -> cross-attention to visual features
text hidden states -> MLP
```

优点：

1. 视觉信息通过专门 cross-attention 注入。
2. 可以处理多图和上下文交错。
3. 对冻结大语言模型比较友好。

缺点：

1. 架构改动更大。
2. 实现和训练复杂。
3. 插入 cross-attention 层会增加参数和计算。

Flamingo 还使用 Perceiver Resampler 把大量视觉特征压缩成固定数量的 visual tokens，降低后续计算成本。

## 4.6 Perceiver Resampler

Vision encoder 输出的 patch tokens 可能很多。Perceiver Resampler 的目标是用一组固定数量的 latent queries 从视觉特征中提取信息。

直觉：

```text
many visual patch tokens -> fixed number of visual latents
```

例如：

```text
1024 visual tokens -> 64 visual latents
```

好处：

1. 控制视觉 token 数。
2. 降低 LLM 侧计算。
3. 便于处理不同分辨率或多图输入。

代价：

1. 可能压缩掉细节。
2. 需要额外训练模块。
3. 对 OCR 和小目标可能不利。

## 4.7 BLIP-2 和 Q-Former

BLIP-2 的核心模块是 Q-Former。Q-Former 使用一组 learnable query tokens 去查询冻结的 vision encoder 输出，从而得到少量和语言相关的视觉表示。

结构：

```text
image -> frozen vision encoder -> image features
learnable queries -> Q-Former attends to image features -> query outputs
query outputs -> LLM
```

Q-Former 的作用：

1. 从大量视觉特征中提取和语言相关的信息。
2. 降低视觉 token 数。
3. 作为 frozen vision encoder 和 frozen LLM 之间的桥。

优点：

1. 参数效率较高。
2. 可以冻结大模型主体。
3. 显著减少送入 LLM 的视觉 token。

缺点：

1. 架构更复杂。
2. query 数量限制可能造成信息瓶颈。
3. 对细粒度视觉任务需要仔细调优。

## 4.8 Connector 的几种选择

Connector 是视觉和语言之间的桥。

常见选择：

1. Linear projector。
2. MLP projector。
3. Q-Former。
4. Perceiver Resampler。
5. Cross-attention adapter。

选择依据：

1. 是否需要压缩视觉 token。
2. 是否冻结 vision encoder 和 LLM。
3. 任务是否需要细粒度信息。
4. 训练数据规模。
5. 延迟和显存预算。

简单项目中，MLP projector 足够常见。大规模、多图、高分辨率或视频场景中，resampler 或 cross-attention 可能更合适。

## 4.9 VLM 的训练阶段

很多 VLM 训练会分阶段。

### 4.9.1 对齐预训练

目标：让视觉 tokens 能被 LLM 初步理解。

常见做法：

1. 冻结 vision encoder。
2. 冻结或部分冻结 LLM。
3. 训练 projector。
4. 使用图文 caption 数据。

### 4.9.2 多模态 Instruction Tuning

目标：让模型学会按用户指令回答图像问题。

数据包括：

1. 图片描述。
2. VQA。
3. OCR QA。
4. 图表问答。
5. 多轮图文对话。
6. 拒答和安全样本。

### 4.9.3 高分辨率和专项增强

针对 OCR、文档、图表、小目标等能力，可能继续加入专项数据和高分辨率训练。

## 4.10 冻结策略

VLM 训练时不一定全参数训练。

常见策略：

1. 冻结 vision encoder，只训练 projector。
2. 冻结 LLM，只训练 connector。
3. 冻结大部分参数，用 LoRA 微调 LLM。
4. 全参微调整个 VLM。

取舍：

1. 冻结更多参数更省显存、更稳定，但能力上限可能受限。
2. 解冻更多参数适应性更强，但训练成本和灾难性遗忘风险更高。
3. LoRA 是常见折中方案。

面试中要能解释：训练 projector 只是让视觉特征进入语言空间，不一定足以获得复杂视觉推理能力。

## 4.11 多图输入

VLM 可能需要处理多张图片。

例如：

```text
<image_1> <image_2>
比较这两张图片有什么不同。
```

难点：

1. 每张图都有多个 visual tokens。
2. 多图 token 占用上下文。
3. 模型要知道哪个 token 属于哪张图。
4. 多图对比需要跨图关系建模。

常见做法：

1. 给每张图加特殊分隔 token。
2. 使用位置或图片 id embedding。
3. 限制图片数量和分辨率。
4. 对视觉 tokens 做压缩。

## 4.12 VLM 中的 attention 成本

如果把 visual tokens 直接拼进 LLM，上下文长度会变长。

假设：

```text
text tokens = 512
visual tokens = 1024
total tokens = 1536
```

LLM self-attention 成本随总 token 数增长很快。

这就是为什么 VLM 很关注：

1. 图像分辨率。
2. patch size。
3. visual token compression。
4. resampler。
5. dynamic resolution。
6. token pruning。

## 4.13 VLM 和 OCR/文档理解

普通 VLM 不一定擅长 OCR 和文档理解。

原因：

1. 图片 resize 后小字模糊。
2. visual tokens 太少，文字细节丢失。
3. 训练数据缺少文档、表格和截图。
4. 模型容易根据语言先验猜答案。

改进方向：

1. 高分辨率输入。
2. 图片切块。
3. OCR 专项训练数据。
4. 外部 OCR 工具结合。
5. 文档 layout-aware 设计。

## 4.14 VLM 的输出和 loss

VLM 通常仍然用语言模型的 next-token prediction loss 训练回答文本。

样本：

```text
User: <image> 这张图里有几只狗？
Assistant: 有两只狗。
```

训练时：

1. 图片通过 vision encoder 得到 visual tokens。
2. 文本通过 tokenizer 得到 text tokens。
3. visual tokens 和 text tokens 组合成 LLM 输入。
4. labels 只对 assistant 回答部分计算 loss。

这说明 VLM 的训练目标通常仍是语言生成 loss，只是条件中多了视觉信息。

## 4.15 常见 VLM 架构对比

### LLaVA 风格

优点：简单、直接、容易实现。

缺点：视觉 token 占上下文，细粒度和高分辨率成本高。

### Flamingo 风格

优点：cross-attention 注入视觉信息，适合多图和交错输入。

缺点：架构复杂，训练成本高。

### BLIP-2 / Q-Former 风格

优点：通过 query 压缩视觉特征，参数效率高。

缺点：query 可能成为信息瓶颈。

### Resampler 风格

优点：固定视觉 token 数，控制成本。

缺点：压缩可能损失细节。

## 4.16 VLM 常见失败模式

1. 视觉幻觉：图里没有的物体被说出来。
2. OCR 错误：文字读错或漏读。
3. 计数错误：小物体数量不准。
4. 空间关系错误：左右、上下、包含关系判断错。
5. 细粒度识别错误：品种、型号、图标识别不准。
6. 图表理解错误：读数、趋势和单位混淆。
7. 多图混淆：把不同图片内容混在一起。

这些失败模式往往和 vision encoder、分辨率、视觉 token 压缩、训练数据和语言先验有关。

## 4.17 面试官会怎么问

### 问题一：VLM 的基本架构是什么？

回答模板：

```text
常见 VLM 由 vision encoder、connector 和 LLM 组成。图片先经过 vision encoder 得到 visual tokens，再通过 projector、Q-Former、resampler 或 cross-attention 适配到 LLM，最后 LLM 结合文本 prompt 和视觉信息生成回答。
```

### 问题二：LLaVA 风格 VLM 是怎么接入图片的？

回答模板：

```text
LLaVA 风格通常用 CLIP vision encoder 提取 patch-level visual tokens，再用 MLP projector 映射到 LLM hidden size，然后把 visual tokens 插入到文本 token 序列中，让 LLM 通过 self-attention 同时处理视觉和文本上下文。
```

### 问题三：Q-Former 的作用是什么？

回答模板：

```text
Q-Former 使用一组 learnable query tokens 去查询 vision encoder 输出，从大量视觉特征中提取少量和语言相关的视觉表示。它是 frozen vision encoder 和 LLM 之间的桥，能减少送入 LLM 的视觉 token 数。
```

### 问题四：为什么 VLM 需要 projector？

回答模板：

```text
因为 vision encoder 输出维度通常和 LLM hidden size 不一致，而且视觉特征空间和语言模型表示空间也不同。Projector 负责维度对齐和表示适配，让 visual tokens 能作为 LLM 的输入 embedding 使用。
```

### 问题五：VLM 为什么会视觉幻觉？

回答模板：

```text
原因可能包括 vision encoder 没捕捉到细节、视觉 token 压缩损失信息、训练数据中语言先验太强、LLM 倾向补全高概率文本，以及缺少基于视觉证据的约束。模型可能生成流畅答案，但答案不被图片支持。
```

## 4.18 小练习

1. 画出 vision encoder + projector + LLM 的 VLM 数据流。
2. 比较 LLaVA、Flamingo、BLIP-2 三类架构。
3. 解释 image token 和 visual tokens 的区别。
4. 设计一个多图输入模板。
5. 计算 `1024` 个 visual tokens 拼接到 `512` 个 text tokens 后的总 token 数。
6. 列出 5 种 VLM 视觉幻觉 bad case。
7. 说明为什么 OCR 场景可能需要高分辨率或外部工具。

## 4.19 本章总结

VLM 的本质是让 LLM 能使用视觉信息。主流架构都围绕一个问题展开：如何把 vision encoder 输出的视觉特征接入语言模型。

需要记住：

1. 最常见结构是 vision encoder + connector + LLM。
2. LLaVA 风格用 projector 把 visual tokens 拼进 LLM 输入。
3. Flamingo 用 cross-attention 注入视觉信息。
4. BLIP-2 用 Q-Former 从视觉特征中抽取少量 query 表示。
5. Perceiver Resampler 用固定数量 latents 压缩视觉 tokens。
6. VLM 训练通常包括视觉语言对齐预训练和多模态 instruction tuning。
7. 高分辨率、OCR、多图和视觉幻觉是 VLM 工程中的关键难点。

下一章会进入多模态 instruction tuning，重点讲图文对话数据格式、chat template、image token、assistant-only loss mask、多轮样本和安全拒答数据。
