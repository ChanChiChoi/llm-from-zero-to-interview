# 第五章：多模态 Instruction Tuning

多模态 instruction tuning 的目标，是让模型学会按照用户指令使用图像、截图、文档、图表、视频帧或语音信息来回答问题。前面讲的 VLM 架构解决了“视觉特征如何接入 LLM”，但接入之后模型并不会自动知道如何遵循多模态指令。它需要通过高质量图文对话数据训练，学习什么时候看图、如何回答、什么时候拒答、如何处理资料不足、如何避免视觉幻觉。

本章重点讲多模态 SFT 的数据格式、image token、chat template、assistant-only loss mask、OCR、图表理解、grounding、多轮图文对话、数据质量和评估思路。

## 5.1 为什么需要多模态 Instruction Tuning

一个 VLM 即使有 vision encoder 和 LLM，也可能只会把图片特征接入语言模型，但不会稳定完成用户任务。

例如用户问：

```text
<image>
这张图里的交通灯是什么颜色？
```

模型需要做到：

1. 理解 `<image>` 对应图片。
2. 找到图片中的交通灯。
3. 判断颜色。
4. 用自然语言回答。
5. 不编造看不到的内容。

这些行为需要 instruction tuning 训练。

面试回答：

```text
多模态 instruction tuning 的作用是让模型学会按照用户指令使用视觉信息。架构上把图片接入 LLM 只是第一步，SFT 数据会教模型如何回答图片问答、OCR、图表、截图、多轮对话和拒答边界，并通过 assistant-only loss mask 只训练模型生成回答部分。
```

## 5.2 一个多模态 SFT 样本长什么样

最简单样本：

```json
{
  "image": "images/cat_001.jpg",
  "messages": [
    {"role": "user", "content": "<image> 请描述这张图片。"},
    {"role": "assistant", "content": "图片中有一只猫坐在窗边。"}
  ]
}
```

关键字段：

1. 图片路径或图片对象。
2. 对话 messages。
3. `<image>` 占位符。
4. assistant 回答。

多图样本可能是：

```json
{
  "images": ["before.jpg", "after.jpg"],
  "messages": [
    {"role": "user", "content": "<image_1><image_2> 比较这两张图的变化。"},
    {"role": "assistant", "content": "第二张图中桌面比第一张更整洁，杯子被移到了右侧。"}
  ]
}
```

训练代码必须保证图片数量、占位符数量和视觉 tokens 插入位置一致。

## 5.3 Image Token 和 Chat Template

VLM 通常使用特殊 token 表示图片位置，例如：

```text
<image>
```

但不同模型的 chat template 可能不同。

例如：

```text
<|user|>
<image>
请描述这张图片。
<|assistant|>
这是一张城市夜景照片。
```

或者：

```text
USER: <image>
What is in the image?
ASSISTANT: There is a dog on the beach.
```

模板一致性非常重要。如果训练和推理使用不同模板，模型可能不知道图片在哪里，也可能输出格式混乱。

常见检查项：

1. `<image>` 是否出现在 user 消息中。
2. tokenizer 是否包含对应 special token。
3. 图片 token 展开数量是否和模型配置一致。
4. assistant 起始标记是否正确。
5. 多轮对话分隔符是否一致。

## 5.4 Assistant-only Loss Mask

和文本 SFT 一样，多模态 SFT 通常只训练 assistant 回答部分。

不应该训练模型去预测：

1. system prompt。
2. user prompt。
3. `<image>` 占位符。
4. visual tokens。
5. padding token。

labels 通常这样设计：

```text
input_ids: [system tokens][user tokens][image marker][assistant tokens]
labels:    [-100 ...    ][-100 ... ][-100       ][assistant token ids]
```

`-100` 表示忽略 loss。

如果把 user prompt 或 image token 也算进 loss，模型会被训练去复述用户输入或预测特殊占位符，优化目标会偏掉。

面试回答：

```text
多模态 SFT 中通常只对 assistant 回复计算 loss。system、user、image token、visual tokens 和 padding 都应该 mask 成 -100。这样训练目标才是让模型基于图像和指令生成正确回答，而不是复述输入或预测图片占位符。
```

## 5.5 图像描述数据

Image caption 数据是最基础的多模态数据。

样本：

```json
{
  "image": "street.jpg",
  "messages": [
    {"role": "user", "content": "<image> 请描述这张图片。"},
    {"role": "assistant", "content": "一条城市街道上有多辆汽车，路边有行人和商店。"}
  ]
}
```

作用：

1. 建立基础视觉描述能力。
2. 学习常见物体、场景和属性。
3. 让模型把视觉信息转成自然语言。

局限：

1. caption 往往偏粗粒度。
2. 不一定训练问答和推理。
3. 可能包含主观描述或遗漏细节。

## 5.6 VQA 数据

VQA 训练模型根据图片回答具体问题。

样本：

```json
{
  "image": "kitchen.jpg",
  "messages": [
    {"role": "user", "content": "<image> 桌子上有几个杯子？"},
    {"role": "assistant", "content": "桌子上有三个杯子。"}
  ]
}
```

VQA 数据覆盖：

1. 物体识别。
2. 属性识别。
3. 计数。
4. 空间关系。
5. 动作和事件。
6. 常识推理。

注意：计数和空间关系是 VLM 常见弱点，数据构造时要特别加入这类样本。

## 5.7 OCR 数据

OCR 数据训练模型读取图片中的文字。

样本：

```json
{
  "image": "receipt.jpg",
  "messages": [
    {"role": "user", "content": "<image> 这张小票上的总金额是多少？"},
    {"role": "assistant", "content": "小票上的总金额是 128.50 元。"}
  ]
}
```

OCR 场景包括：

1. 截图文字。
2. 小票和发票。
3. 文档扫描件。
4. 路牌和招牌。
5. 表格和图表文字。

难点：

1. 字小。
2. 图片模糊。
3. 倾斜和遮挡。
4. 多语言混杂。
5. 版面结构复杂。

训练 OCR 能力时，常常需要高分辨率、切图、OCR 专项数据或外部 OCR 工具辅助。

## 5.8 图表理解数据

图表理解比 OCR 更进一步。模型不仅要读文字和数字，还要理解坐标、趋势、图例和单位。

样本：

```json
{
  "image": "sales_chart.png",
  "messages": [
    {"role": "user", "content": "<image> 哪个季度销售额最高？"},
    {"role": "assistant", "content": "第四季度销售额最高。"}
  ]
}
```

图表任务包括：

1. 读取数值。
2. 比较大小。
3. 判断趋势。
4. 理解图例。
5. 单位换算。
6. 多步推理。

常见错误：

1. 把坐标轴读错。
2. 忽略单位。
3. 混淆图例颜色。
4. 用语言先验猜趋势。

## 5.9 Grounding 数据

Grounding 是把语言描述和图像区域对应起来。

例如：

```text
指出图中红色汽车的位置。
```

输出可以是：

1. bounding box。
2. mask。
3. 区域描述。
4. 引用编号。

样本可能包含：

```json
{
  "image": "parking.jpg",
  "question": "红色汽车在哪里？",
  "answer": "红色汽车位于图片左下方。",
  "bbox": [34, 210, 180, 320]
}
```

Grounding 能力有助于减少视觉幻觉，因为模型需要把回答对应到具体视觉证据。

## 5.10 多轮图文对话

多轮对话要求模型记住上下文和图片引用。

示例：

```json
{
  "image": "room.jpg",
  "messages": [
    {"role": "user", "content": "<image> 这是什么房间？"},
    {"role": "assistant", "content": "这是一个客厅。"},
    {"role": "user", "content": "沙发是什么颜色？"},
    {"role": "assistant", "content": "沙发是灰色的。"}
  ]
}
```

难点：

1. 第二轮可能省略图片，但仍依赖前图。
2. 代词和上下文引用要正确。
3. 多图多轮时容易混淆图片。
4. loss mask 要正确覆盖多个 assistant 回复。

## 5.11 拒答和资料不足数据

VLM 不应该总是回答。图片信息不足时，应说明无法确定。

样本：

```json
{
  "image": "blurry_plate.jpg",
  "messages": [
    {"role": "user", "content": "<image> 车牌号是多少？"},
    {"role": "assistant", "content": "图片中的车牌过于模糊，我无法可靠识别车牌号。"}
  ]
}
```

这类数据很重要，因为多模态幻觉经常来自模型在视觉证据不足时硬答。

拒答边界要清楚：

1. 看不清就说看不清。
2. 资料不足就说明不足。
3. 高风险场景不要过度自信。
4. 不要编造图片中不存在的信息。

## 5.12 数据质量检查

多模态数据质量比文本更难检查。

需要检查：

1. 图片是否能打开。
2. 图片和对话是否匹配。
3. `<image>` 数量是否正确。
4. 回答是否真的被图片支持。
5. OCR/图表答案是否准确。
6. 是否存在重复样本。
7. 是否包含有害内容。
8. 是否存在隐私信息。
9. 多轮对话是否上下文一致。

自动检查可以做格式和一致性，关键样本仍需要人工抽检。

## 5.13 多模态数据混合

训练 VLM 不能只用一种数据。

常见 mixture：

1. Caption 数据。
2. VQA 数据。
3. OCR 数据。
4. 图表数据。
5. 文档问答。
6. 多轮对话。
7. 安全拒答。
8. 通用纯文本指令数据。

为什么要保留纯文本数据？因为多模态微调可能损害 LLM 原有语言能力。加入一定比例纯文本数据有助于减少语言能力回退。

数据配比会影响模型能力倾向。例如 OCR 数据太少，模型会更依赖语言先验；安全数据太强，可能 over-refusal。

## 5.14 训练时的 batch 组织

多模态 batch 比文本 batch 更复杂。

batch 里可能包含：

```python
{
    "input_ids": ...,          # [B, T]
    "labels": ...,             # [B, T]
    "attention_mask": ...,     # [B, T]
    "pixel_values": ...,       # [B, 3, H, W]
    "image_sizes": ...,        # original sizes
}
```

多图场景可能是：

```python
{
    "input_ids": ...,
    "labels": ...,
    "pixel_values": [image tensors per sample],
    "num_images": ...,
}
```

难点：

1. 图片尺寸可能不同。
2. 每个样本图片数量不同。
3. 文本长度不同。
4. visual tokens 插入后序列长度变化。
5. padding 和 label mask 要一起对齐。

## 5.15 常见训练 bug

### 5.15.1 Image token 和图片数量不一致

表现：模型报错或视觉 tokens 插入错位。

排查：统计每个样本 `<image>` 数量和图片数量。

### 5.15.2 labels 没有正确 mask

表现：loss 异常，模型学会复述 user prompt，或输出特殊 token。

排查：打印 input_ids decode 和 labels decode，确认只有 assistant 部分参与 loss。

### 5.15.3 图片预处理不一致

表现：训练或推理效果明显差。

排查：确认 resize、normalize、processor 和 vision encoder 预训练配置一致。

### 5.15.4 OCR 样本答案错误

表现：模型学习错误文字，评估混乱。

排查：对 OCR/图表样本做更严格人工抽检。

## 5.16 评估多模态 SFT 是否有效

评估不能只看训练 loss。

应覆盖：

1. 普通图片描述。
2. 物体识别。
3. 计数。
4. 空间关系。
5. OCR。
6. 图表理解。
7. 多轮对话。
8. 资料不足拒答。
9. 视觉幻觉。
10. 纯文本能力回归。

常用指标：

1. Accuracy。
2. Exact Match。
3. LLM-as-Judge。
4. 人工评测。
5. Hallucination rate。
6. Abstention accuracy。

## 5.17 面试官会怎么问

### 问题一：多模态 instruction tuning 的目标是什么？

回答模板：

```text
目标是让模型学会按照用户指令使用视觉信息。它不仅训练模型描述图片，还训练 VQA、OCR、图表、多轮对话、拒答和安全边界。训练时通常只对 assistant 回复计算 loss，图片 token、user prompt 和 padding 都要 mask 掉。
```

### 问题二：多模态 SFT 样本包含哪些字段？

回答模板：

```text
通常包含图片路径或 pixel values、messages 对话、image token 占位符和 assistant 回复。训练时还会构造 input_ids、labels、attention_mask、pixel_values 等字段，并确保图片数量和 image token 数量一致。
```

### 问题三：为什么只对 assistant 部分算 loss？

回答模板：

```text
因为训练目标是让模型基于图片和用户指令生成助手回答，而不是预测用户输入、system prompt 或 image token。对非 assistant 部分算 loss 会让模型学会复述输入或预测占位符，干扰指令遵循。
```

### 问题四：OCR 数据为什么重要？

回答模板：

```text
很多真实多模态任务来自截图、文档、小票、图表和路牌，都依赖读取图片中的文字。普通 caption 和 VQA 数据不足以训练稳定 OCR 能力，因此需要专门 OCR 数据、高分辨率处理或外部 OCR 工具结合。
```

### 问题五：如何减少 VLM 幻觉？

回答模板：

```text
可以从数据和评估两侧减少幻觉。数据上加入资料不足拒答、grounding、细粒度 VQA 和高质量人工标注；训练时确保 loss mask 正确；评估时做 claim-level 或视觉证据检查，要求模型在看不清或证据不足时明确说明无法确定。
```

## 5.18 小练习

1. 写一个单图多模态 SFT JSON 样本。
2. 写一个多图对比样本。
3. 给一段 user/assistant 对话设计 labels mask。
4. 构造 3 个 OCR 问答样本。
5. 构造 3 个图表理解样本。
6. 构造 3 个资料不足必须拒答的样本。
7. 设计一个多模态 SFT 数据质量 checklist。
8. 设计一个 VLM SFT 后的评估表。

## 5.19 本章总结

多模态 instruction tuning 是让 VLM 从“能接收图片”变成“能按指令使用图片”的关键阶段。它的核心不只是增加图像输入，而是构造高质量图文对话数据，并正确处理 image token、chat template、assistant-only loss mask、多轮对话、OCR、图表、grounding 和拒答边界。

需要记住：

1. 图片和 `<image>` 占位符必须严格对齐。
2. 训练 loss 通常只覆盖 assistant 回复。
3. OCR、图表、计数、空间关系是 VLM 常见短板。
4. 多轮图文对话需要处理上下文引用。
5. 拒答数据能减少视觉证据不足时的幻觉。
6. 数据质量和 mixture 决定模型能力分布。
7. 评估要覆盖视觉理解、OCR、图表、幻觉、安全和文本能力回归。

下一章会进入 diffusion 基础，讲清加噪、去噪、噪声预测、采样过程和为什么 diffusion 能生成图像。
