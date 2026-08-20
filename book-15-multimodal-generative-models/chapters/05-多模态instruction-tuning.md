# 第五章：多模态 Instruction Tuning——把看见变成可监督行为

上一章讨论了图片如何经过 vision encoder、connector 和语言模型，进入一次生成计算。本章继续追问一个更容易被忽略的问题：模型已经“能接收图片”之后，为什么还要训练多模态 instruction tuning？

因为接口接通不等于行为学会。一个模型可以成功读取像素、完成一次前向计算，却仍然可能忽略图片，复述用户问题，把图片里的数字看错，混淆多张图片，或者在证据不足时自信地编造答案。Instruction tuning 的作用，是用带有任务意图、媒体绑定、回答责任和证据边界的样本，把这些行为变成可以优化、可以审计、可以评估的监督目标。

本章把多模态 SFT 当作一门数据与训练工程来讲。读者会看到：

- 一条样本如何同时描述媒体、对话、任务、证据和风险；
- chat template 如何把消息编译成模型真正看到的序列；
- image reference 为什么不是一个普通字符串；
- assistant-only label mask 如何与 attention mask 区分；
- caption、VQA、OCR、图表、grounding、多图、视频、音频和拒答数据各自教什么；
- 数据支持率、重复污染、隐私、配比和有效 token 如何测量；
- 训练 batch、动态分辨率、梯度累积和冻结策略怎样影响监督；
- 如何设计一套不能被总体平均数掩盖的评估；
- 如何用一个标准库 Python demo 检查样本契约，但不把格式检查误认为视觉理解能力。

## 0. 研究对象、资料与阅读方法

### 0.1 本章的边界

多模态 instruction tuning 通常指使用图文、视频文、音频文或文档问答等指令样本，对已经具备基础表示能力的模型进行监督微调。这里的“监督”不只意味着有一段答案，还意味着训练系统知道：

1. 哪些媒体属于当前样本；
2. 用户的问题在哪里；
3. assistant 的输出从哪里开始、到哪里结束；
4. 哪些答案是由媒体证据支持的；
5. 哪些样本要求模型说明不确定或拒绝回答；
6. 这些字段应该如何进入 tokenizer、collator 和 loss。

本章不重新推导视觉编码器、projector 或 cross-attention 的内部结构；那些结构决定信息怎样流动，而本章研究数据怎样规定模型应该做什么。本章也不把一个公开论文中的训练配方直接当成所有产品都适用的配方。不同模型的 tokenizer、placeholder、视觉 token 展开方式、上下文上限、图像预处理和冻结策略都可能不同。

### 0.2 资料层级与证据边界

| 资料层级 | 可以支持的判断 | 不能直接支持的判断 |
| --- | --- | --- |
| LLaVA、InstructBLIP、MiniGPT-4 等原始论文 | 论文所述的数据构造、训练阶段和实验协议 | 当前产品的全部数据、全部安全行为和真实业务成功率 |
| TextVQA、ChartQA 等任务论文 | 任务定义、标注形式和公开评估协议 | 模型在任意扫描件或任意语言上的 OCR 能力 |
| Transformers、PyTorch 等官方文档 | 某个版本 API、模板或 loss 的语义 | 一个自定义 collator 一定正确 |
| 开源仓库 | 特定 commit 的实现细节 | 实现之外的部署路径和未来版本行为 |
| 本章 demo | schema、计数、预算和 mask 的教学审计 | 图片是否真的被理解、答案是否真的有视觉依据 |

阅读论文时，先区分“论文报告了什么”与“论文没有测量什么”。例如，论文在某个 VQA benchmark 上取得较好分数，说明它在该数据、该模型版本和该评价协议下有效；这并不能自动证明它能读清用户上传的低清合同，也不能证明它在调用工具之前会遵守权限边界。

### 0.3 初学者和专家各自要抓住什么

初学者可以先把一条多模态样本想成一张责任表：

- 图片是条件；
- 用户消息是任务；
- assistant 消息是要学习的输出；
- 证据标注说明答案为什么成立；
- refusal 或 uncertainty 标注说明什么时候不能硬答。

专家则要进一步记录数据的版本、来源、许可、媒体 hash、模板版本、token 长度、label 分母、任务采样概率、切分键和评估切片。只记录一段 JSON，而不记录这些上下文，未来很难解释一次 loss 下降究竟代表能力提升、数据泄漏，还是监督口径发生了变化。

### 0.4 本章要回答的四个问题

第一，样本如何从“有图有答案”变成有明确输入输出边界的训练对象？

第二，模板、placeholder、视觉 token 和 label mask 如何协同，而不是各自“看起来合理”？

第三，不同任务的数据为什么不能互相替代？caption 能教什么，OCR 又缺什么？

第四，如何把质量、配比和评估写成可复盘的量，而不是训练结束后凭感觉评价模型？

## 1. 为什么接通视觉输入仍然不够

### 1.1 架构能力与行为能力是两回事

设媒体集合为 M，用户问题和对话历史为 U，assistant 回答为 A。一个生成式 VLM 最终要学习的是条件分布：

~~~math
p_\theta(A\mid M,U)
=\prod_{t=1}^{T_A}
p_\theta(a_t\mid a_{<t},M,U).
~~~

视觉 encoder 和 connector 主要负责把 M 变成语言模型能够访问的表示；instruction tuning 则通过样本告诉模型，在什么 U 下应该生成什么 A。

如果只有图片-文本预训练或简单的视觉表示对齐，模型可能学会“图片和文字大致相关”，却不一定学会：

- 按用户指定的字段回答；
- 从三张图片中区分第一张和第二张；
- 只报告图片能支持的数字；
- 在问题要求 JSON 时输出稳定结构；
- 在图片模糊时说明无法确定。

因此，instruction tuning 不是给 connector 再加一个名字，而是给条件生成过程加入任务、格式和责任。

### 1.2 三种常见的“模型看起来会了”假象

第一种假象是语言先验。问题是“图片中有一只狗吗”，即使图片被替换成一张没有狗的图片，模型也可能因为训练语料中的常见模式而回答“有”。如果答案对关键图片替换不敏感，流畅度不能证明视觉使用。

第二种假象是主题正确、细节错误。模型可能知道图片是一张发票，却把金额、日期或税率写错。全局语义对了，不代表细粒度 OCR 和数字比较对了。

第三种假象是格式正确、证据错误。模型可以输出一个合法 JSON，也可以填入页码和坐标，但这些字段可能不是从正确页面和正确区域得到的。结构化输出只解决协议，不自动解决事实支持。

### 1.3 SFT 在这里到底优化什么

对一批样本 D，最简单的 token-level 监督目标是：

~~~math
\mathcal{L}_{\mathrm{SFT}}
=
-\frac{1}{\sum_{i,t}m_{i,t}}
\sum_{i=1}^{N}\sum_{t=1}^{T_i}
m_{i,t}
\log p_\theta(y_{i,t}\mid y_{i,<t},M_i,U_i).
~~~

这里：

- N 是样本数；
- y_{i,t} 是第 i 条样本的目标 token；
- m_{i,t}=1 表示该位置需要计算监督，m_{i,t}=0 表示忽略；
- 分母是有效监督 token 数，而不是 padded 序列长度。

这个目标教的是“在给定媒体和指令的条件下生成目标序列”。它不自动教会模型辨别所有图片细节，也不自动提供 grounding、拒答、隐私和工具安全能力。若训练答案本身没有页码、区域、数值或不确定性信息，模型不会凭空从 loss 中得到这些责任。

### 1.4 instruction tuning 不是数据清洗的同义词

数据清洗解决“样本是否可用”，instruction tuning 解决“可用样本规定什么行为”。两者相互依赖，但不能混成一句“把数据清理干净”。

例如：

- 图片路径失效，是媒体完整性问题；
- 图片与回答不匹配，是语义支持问题；
- user token 被算入 label，是监督边界问题；
- 所有数据都来自 caption，是任务覆盖问题；
- 车牌号样本没有模糊拒答，是不确定性设计问题。

这些问题会在不同阶段暴露，应该有不同的字段和指标。

## 2. 把样本定义成多模态文档

### 2.1 从字符串升级为结构化对象

最简单的旧式样本常把所有内容写在一条字符串中：

~~~json
{
  "image": "images/cat_001.jpg",
  "text": "<image> 请描述这张图片。"
}
~~~

这种表示可以作为教学起点，但在多图、多轮和文档任务中很快不够用。更稳健的样本应区分媒体对象、消息内容、任务标签和证据标注：

~~~json
{
  "sample_id": "contract_017",
  "media": [
    {
      "media_id": "page_1",
      "type": "image",
      "uri": "contracts/017/page_1.png",
      "sha256": "example-hash",
      "width": 2048,
      "height": 2896
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "media_id": "page_1"},
        {"type": "text", "text": "付款期限是多少？请给出页码和原文区域。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "付款期限为收到发票后 60 天。"}
      ]
    }
  ],
  "task": "document_qa",
  "annotations": {
    "claims": [
      {
        "claim_id": "c1",
        "text": "付款期限为收到发票后 60 天。",
        "supported_by": ["e1"]
      }
    ],
    "evidence": [
      {
        "evidence_id": "e1",
        "media_id": "page_1",
        "bbox": [420, 1330, 1510, 1450],
        "transcript": "付款期限：收到发票后60天。"
      }
    ]
  },
  "policy": {
    "may_answer": true,
    "sensitive": false
  }
}
~~~

这里的 content 是一个列表，而不是把图片和文本都塞进一个字符串。这样做的好处是：

1. 图片对象有自己的 id、尺寸、hash 和来源；
2. 同一张图可以在多轮中被引用，而不必复制文件；
3. 文本和媒体的相对顺序可以明确保存；
4. 证据可以指向具体页面和区域；
5. collator 可以在编译前发现引用缺失。

初学者可以先记住“对象、消息、答案、证据”四层。专家还要把 processor 版本、模板版本、许可、去重键、人工审核状态和切分集合写入 artifact 元数据。

### 2.2 单图样本、多图样本与交错媒体

在每个媒体对象只被引用一次的简化 schema 中，单图问答的最小不变量是：

~~~math
n_{\mathrm{media}}(s)
=
n_{\mathrm{refs}}(s),
~~~

其中 n_media 是样本中实际绑定的媒体对象数，n_refs 是消息内容中被引用的媒体数。一般 schema 还必须满足每个引用 id 都能在 media 表中找到；对一张图重复引用两次，n_refs 可以是 2，但这不等于一定需要加载两份图像，数据模型要进一步保存引用关系和是否复用同一视觉表示。

多图比较不能只保存：

~~~json
{
  "images": ["before.jpg", "after.jpg"],
  "prompt": "<image_1><image_2> 哪一张更整洁？"
}
~~~

还要规定 image_1 和 image_2 的顺序、id 和回答中的指代。否则“第一张”“第二张”“左图”“右图”在不同预处理或排序逻辑下可能错位。

交错图文样本可以写成：

~~~json
{
  "media": [
    {"media_id": "chart_a", "type": "image", "uri": "a.png"},
    {"media_id": "chart_b", "type": "image", "uri": "b.png"}
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "先看第一张图的销售趋势："},
        {"type": "image", "media_id": "chart_a"},
        {"type": "text", "text": "再看第二张图的利润趋势："},
        {"type": "image", "media_id": "chart_b"},
        {"type": "text", "text": "比较两个趋势。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "第一张图上升更快，第二张图在末期出现回落。"}
      ]
    }
  ]
}
~~~

如果系统把所有图片无条件移到 prompt 最前面，视觉 token 的数量可能仍然正确，但叙事顺序和归因关系已经改变。

### 2.3 视频和音频也要有对象边界

视频样本不能只保存一个视频文件名。至少要记录采样策略：

~~~json
{
  "media_id": "meeting_03",
  "type": "video",
  "uri": "meeting_03.mp4",
  "sampled_frames": [0, 48, 96, 144],
  "fps": 2
}
~~~

音频样本也要区分原始音频、转写文本和时间区间：

~~~json
{
  "media_id": "call_07",
  "type": "audio",
  "uri": "call_07.wav",
  "transcript_segments": [
    {"start": 12.4, "end": 16.8, "text": "付款日期改到月底。"}
  ]
}
~~~

如果训练目标只看转写文本，模型学到的可能是语言问答，而不是从声学信号中识别说话人、语气或非语言事件。反过来，如果视频帧采样过稀，模型无法从监督中恢复未采样的动作。媒体采样策略本身就是样本语义的一部分。

### 2.4 答案、证据与策略标签

一段自然语言答案至少可能包含三种不同信息：

1. 对媒体内容的事实陈述；
2. 对问题的组织和解释；
3. 关于不确定性、隐私或权限的策略判断。

把这三类信息混在一个字符串中，会让后续评估很困难。更好的做法是让 annotations 保存 claim 和 evidence，让 policy 保存拒答、敏感和可回答状态。

例如“第 2 页的金额是 1,280 元，但图片分辨率不足以确认收款账户”同时包含一个支持充分的 claim 和一个不确定性判断。训练目标可以保留自然语言答案，但评估应分别检查金额 claim、证据页码和不确定性表达。

### 2.5 来源、许可、隐私与可删除性

多模态数据的风险不只来自文本。图片中可能有姓名、证件号、人脸、车牌、屏幕内容和内部文件；音频中可能包含声音身份和会议内容；视频中还可能包含未被文字标注的旁观者。

一个可复盘的数据 artifact 至少应关联：

- 来源和采集时间；
- 许可或使用依据；
- 原始媒体 hash；
- 脱敏处理版本；
- 人工标注者和审核状态；
- 训练、验证、测试归属；
- 删除请求和下游缓存影响。

hash 主要用于追踪和去重，不等于匿名化。把文件名改成随机字符串，也不等于隐私已经被保护。

## 3. Chat Template 是一个编译器

### 3.1 从消息到模型输入的流水线

chat template 不是为了让日志好看，它把高层消息对象编译成模型可以执行的输入：

~~~text
structured messages
    -> role and content normalization
    -> template rendering
    -> text tokenization
    -> media reference resolution
    -> visual feature insertion or memory construction
    -> input_ids / embeddings / labels / masks
~~~

编译器的输入是消息对象，输出是多个彼此必须一致的张量和 span。只检查最终字符串中出现了 image，就像只检查程序源代码中出现了变量名，不能证明运行时绑定正确。

### 3.2 角色语法与版本

不同模型可能使用不同的 system、user、assistant 标记，也可能在 assistant 结尾添加 generation prompt。训练模板和推理模板若不一致，模型可能出现：

- assistant 起始位置错位；
- 训练时监督了模板符号，推理时却没有这些符号；
- 多轮消息被拼成一轮；
- 末尾的 EOS 或 stop token 不一致；
- user 内容被误当成 assistant 内容。

因此应把模板版本写入数据 artifact，并把“从原始消息得到 assistant span”作为可测试函数。不要用一个正则表达式在所有模型上猜测 assistant 起点。

### 3.3 Image reference 不是一个 token

在很多系统中，消息里的一个 image reference 最终可能展开成 N_v 个连续视觉 embedding；也可能进入独立的视觉 memory，由语言层的 cross-attention 读取。于是：

~~~math
T_{\mathrm{total}}
=
T_{\mathrm{text}}
+
T_{\mathrm{special}}
+
\sum_{j=1}^{I}N_{v,j}
~~~

适用于视觉 token 进入主序列的 prefix 路线。这里：

- I 是图片或视觉片段数量；
- N_{v,j} 是第 j 个片段实际贡献的 token 数；
- T_text 是经过模板后的文本 token 数；
- T_special 是边界、角色和其他特殊 token 数。

如果采用 cross-attention，主文本序列可能不包含全部视觉 token，但视觉 memory、resampler 和 cross-attention 仍然有成本。不能因为 tokenizer 中只有一个 image reference，就把视觉成本记成一个 token。

### 3.4 多图绑定的三个同时成立的条件

对每条样本，至少要同时成立：

1. 消息中的引用 id 存在于 media 表；
2. 解析出的媒体顺序与消息顺序一致；
3. 生成的视觉片段顺序与引用顺序一致。

可以把这三个序列写成：

~~~text
message refs: [page_2, page_5]
media objects: [page_2, page_5]
visual inputs: [features(page_2), features(page_5)]
~~~

任何一层排序都可能改变“第一张图”的含义。尤其是动态 batching、按尺寸排序和多 crop 处理，必须保存从原始 media_id 到视觉片段的映射，而不能只保留一个 batch 内位置。

### 3.5 多轮对话中的图片生命周期

多轮样本有两种不同语义：

- 后续轮次重新附带一张图片；
- 后续轮次引用之前已经上传的图片。

第二种需要会话级 media_id 和权限状态。把图片的文字摘要复制到下一轮，会节省视觉计算，却也可能丢失原始区域证据；把视觉 token 无限复制，则会迅速消耗上下文。工程上可以缓存视觉表示，但缓存键至少要包含媒体 hash、processor 版本、视觉模型版本、分辨率配置和权限状态。

### 3.6 编译结果应该保存哪些 span

为了构造 label mask，编译器最好输出：

~~~json
{
  "input_ids": [101, 102, 103],
  "assistant_spans": [[87, 104], [142, 158]],
  "media_spans": [[22, 598]],
  "role_spans": {
    "system": [[0, 8]],
    "user": [[8, 87], [104, 142]]
  },
  "template_version": "vlm-chat-v3"
}
~~~

数字只是示意。真正重要的是 span 来自编译过程，而不是训练循环再次用字符串搜索猜测。这样可以在样本进入 batch 前检查多轮 assistant span 是否为空、是否重叠、是否超出序列长度。

## 4. Assistant-only Loss Mask：监督责任的边界

### 4.1 输入、标签和 shift

自回归语言模型通常用当前位置的 logits 预测下一个 token。以简化序列为例：

~~~text
input_ids: [BOS, user, image, question, assistant, answer_1, answer_2, EOS]
labels:    [-100, -100, -100, -100, -100, answer_1, answer_2, EOS]
~~~

实际实现还会进行一位 shift。若 logits 的长度为 T，labels 需要和 shift 后的目标位置对齐。最容易犯的错误是把“显示出来的 assistant 起始位置”直接当成 loss 起点，却没有核对 shift 之后的索引。

当连续视觉 embedding 被展开到模型序列后，labels 仍要与完整序列长度对齐；对应视觉 span 的位置应填 `ignore_index`，不能因为视觉输入不是词表 id 就把这些位置从 labels 中悄悄删掉。下面的 toy demo 只统计有效 label 数，但真实 collator 必须同时保存文本 span、视觉 span 和 shift 后的坐标。

### 4.2 为什么视觉条件通常不是 label

本章讨论的连续视觉条件 VLM 中，视觉 embedding 通常参与 hidden state、attention 和最终回答的生成，但不作为词表中的预测目标。把 visual token 放进 labels，可能导致：

- 词表中不存在对应连续向量的目标；
- 模型被迫预测特殊 placeholder；
- assistant 目标的有效分母被稀释；
- 训练日志看似有 loss，实际监督责任已经错位。

这不意味着视觉没有梯度。只要视觉表示参与了 assistant logits，assistant loss 的梯度可以沿着计算图回到 connector，进一步回到被解冻的 vision encoder。若模型采用离散图像 token，并把图像生成也作为自回归目标，则它有另一套词表和监督定义，不能直接套用本节的连续视觉条件规则。

### 4.3 多轮对话需要多个 assistant span

如果一条样本有三轮 user/assistant 交互，三个 assistant 回答都可以是监督目标：

~~~math
m_{i,t}
=
\begin{cases}
1,&t\in\mathcal{S}_{i,\mathrm{assistant}},\\
0,&t\notin\mathcal{S}_{i,\mathrm{assistant}}.
\end{cases}
~~~

这里 S_{i,assistant} 是所有 assistant span 的并集，不是最后一轮 span；实现还要明确这些 span 是 input 坐标还是 shift 后的 label 坐标，并在构造 labels 时完成同一坐标变换。若只监督最后一轮，模型可能学不到前两轮的指代和上下文行为；若把历史 user 内容也设为 1，模型会被鼓励复述对话。

### 4.4 有效分母比 padded shape 更重要

带 mask 的平均 loss 可以写成：

~~~math
\mathcal{L}
=
-\frac{
\sum_{i,t}m_{i,t}\log p_\theta(y_{i,t}\mid \cdots)
}{
\sum_{i,t}m_{i,t}
}.
~~~

这里假设总有效监督 token 数大于零。若一个 batch 的所有样本都没有 assistant span，分母为零，此 batch 没有可解释的 SFT loss，应在 collator 或训练循环处拒绝，而不是把 `NaN` 改成零继续更新。

如果一个 batch 中样本 A 有 100 个 assistant token，样本 B 只有 10 个，简单按样本平均会让短回答和长回答权重相同；按 token 平均则长回答贡献更多。两者都可能合理，但必须明确选择，因为它们对应不同的训练分布。

对于任务加权，可以写成：

~~~math
\mathcal{L}_{\mathrm{weighted}}
=
-\frac{
\sum_{i,t}w_{k(i)}m_{i,t}\log p_\theta(y_{i,t}\mid\cdots)
}{
\sum_{i,t}w_{k(i)}m_{i,t}
},
~~~

其中 k(i) 是样本任务类型，w_k 是任务权重。权重不是越大越好；它会改变模型实际看到的梯度分布。

### 4.5 常见 mask 事故

事故一：user prompt 参与 loss。症状是模型喜欢复述问题、输出固定模板或复制图片说明。

事故二：assistant 起点少一个 token。症状是答案第一个词总是缺失，或者训练和推理的格式边界不同。

事故三：多轮只监督最后一轮。症状是模型不会解析历史指代，问“刚才那张图”时像面对新请求。

事故四：padding 未忽略。症状是 batch 中短样本的 loss 被大量无意义位置污染。

事故五：全样本没有有效 label。训练循环仍可能完成 backward，但分母为零、loss 为 NaN 或梯度没有实际意义。

排查时应同时打印渲染文本、special token、assistant spans、有效 label 数和 shift 后的第一组目标，而不是只看一个总 loss。

## 5. 任务数据：每一种监督教的是不同能力

### 5.1 Caption：从视觉主题到可描述属性

Caption 样本通常是图片配一段描述：

~~~json
{
  "media": [{"media_id": "street_01", "type": "image", "uri": "street.jpg"}],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "media_id": "street_01"},
        {"type": "text", "text": "请描述这张图片。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "一条城市街道上有汽车、行人和商店。"}
      ]
    }
  ],
  "task": "caption"
}
~~~

caption 有三个主要作用：

1. 让模型把视觉表示转成语言；
2. 学习对象、场景、属性和动作的词汇；
3. 提供较宽的图文对齐监督。

它的局限也很明确。许多 caption 只描述显著主题，不保证小字、数量、空间关系和细粒度数字正确；不同标注者可能对“拥挤”“整洁”“开心”等主观属性有不同判断。caption 适合打基础，却不能替代 OCR、图表和 grounding 数据。

高质量 caption 应尽量区分可观察事实和推测。例如“一个人在厨房准备晚餐”可能包含动作推断；若图片只能支持“一个人站在厨房台面旁”，训练数据就不应把未经证实的意图写成事实。对高风险领域，描述粒度和不确定性应有明确规范。

### 5.2 VQA：问题类型决定视觉操作

VQA 不是一个单一能力。下面这些问题需要不同操作：

- 物体识别：图片里有什么；
- 属性识别：物体是什么颜色、形状或状态；
- 计数：有几个对象；
- 空间关系：谁在谁左边、上方或内部；
- 事件理解：发生了什么；
- 组合推理：先找对象，再比较属性。

一个计数样本可以是：

~~~json
{
  "task": "counting",
  "question": "桌子上有几个杯子？",
  "answer": "三个。",
  "evaluation": {"type": "integer", "value": 3}
}
~~~

把数字答案保存成结构化 value 很有用，因为生成字符串“有三个杯子”与“3 个”应当在某些评估中视为同一答案，而“二个”或“八个”不能只靠语言相似度判断。

计数数据还需要反事实和均衡设计。若训练集里“图片中有几只狗”的答案经常是一只，模型可能学到数量先验。可以对同一背景改变对象数量，或交换图片与问题，观察答案是否跟随视觉变化。空间关系则要明确坐标系和关系标签，否则“左边”可能相对于图片、人物还是页面区域产生歧义。

### 5.3 OCR：读取文字不等于理解主题

OCR 样本的监督目标可能有三层：

1. 原样转写；
2. 根据文字回答字段问题；
3. 结合文字和布局进行结构化抽取。

原样转写强调字符、空格、标点和顺序；字段问答强调找到目标字段；结构化抽取还要求键值、表格行列和坐标保持正确。三者混在一个指标里，会掩盖错误类型。

一个票据样本可以写成：

~~~json
{
  "task": "ocr_field",
  "media": [{"media_id": "receipt_01", "type": "image", "uri": "receipt.jpg"}],
  "question": "总金额是多少？",
  "answer": "128.50 元",
  "evidence": {
    "media_id": "receipt_01",
    "bbox": [820, 1710, 1260, 1800],
    "transcript": "TOTAL 128.50"
  }
}
~~~

OCR 的困难包括小字体、模糊、倾斜、遮挡、低对比度、多语言混排和复杂版面。提高输入分辨率只是把更多像素送入视觉塔；如果后续 connector 把细节压缩掉，或语言模型上下文没有预算，放大本身不会自动提升答案。

TextVQA 这类任务专门考查从图片文字中回答问题，适合用来提醒读者：普通 caption 的“这是一张收据”不能证明模型能读出收据金额。文档任务还需要记录页码、行号、字段坐标和归一化规则，避免把一个正确主题判断误报成正确 OCR。

### 5.4 图表：视觉读取、数值运算和语言解释的组合

图表问答至少包含四步：

1. 识别图表类型和坐标轴；
2. 找到图例、类别和单位；
3. 读取或比较数据；
4. 用问题要求的形式解释结果。

例如“第四季度销售额最高”只是一种结论；若任务要求“比第一季度高多少百分比”，模型还必须读取两个数值并完成计算：

~~~math
\Delta_{\%}
=
\frac{v_4-v_1}{v_1}\times 100\%.
~~~

变量 v_1 和 v_4 必须来自正确系列和正确单位。若纵轴单位是千元，答案不能直接当作元；若图例颜色相近，模型可能把两条曲线混在一起。

ChartQA 等公开任务可以支持图表问答的任务定义和评估入口，但不能代替目标业务图表的复测。真实系统常见的错误不是“完全不懂图表”，而是读错刻度、忽略负号、漏掉单位、混淆图例或把趋势猜成数值。数据中应同时包含读值、比较、趋势、单位换算和多步算术，并把中间结构或证据保存下来，便于错误归因。

### 5.5 Grounding：让答案指向区域

Grounding 的目标不是让模型多说一句“在左边”，而是建立文本和空间区域的对应关系。监督可以使用：

- bounding box；
- point；
- segmentation mask；
- 页面区域；
- OCR span；
- 生成答案中的 evidence_id。

用归一化坐标表示一个框时，若图像宽高为 W、H，像素框为 (x_1,y_1,x_2,y_2)，可以写成：

~~~math
\tilde{x}_1=\frac{x_1}{W},\quad
\tilde{y}_1=\frac{y_1}{H},\quad
\tilde{x}_2=\frac{x_2}{W},\quad
\tilde{y}_2=\frac{y_2}{H}.
~~~

这样可以减少不同分辨率的坐标歧义，但仍要说明坐标原点、边界是否包含和图像是否经过 crop。两个系统即使都输出四个数字，只要坐标约定不同，数值就不能直接比较。

最常见的 grounding 误解是把一个全局描述当作区域证据。“这是一份发票”不支持“第 3 行金额是 1280 元”。应按 claim 粒度保存支持关系，并分别评估答案正确和区域正确。

### 5.6 文档问答：结构、页码与证据链

文档图像比自然图片更容易让模型产生“主题正确、字段错误”。训练样本至少要覆盖：

- 页码和文档顺序；
- 标题、段落、表格和脚注；
- 字段和值的绑定；
- 跨页引用；
- 票据、合同和表单中的数字；
- 证据不足时的说明。

结构化答案可以这样表示：

~~~json
{
  "task": "document_qa",
  "question": "付款期限是多少？",
  "answer": {
    "value": "60天",
    "page": 2,
    "bbox": [420, 1330, 1510, 1450],
    "quote": "付款期限：收到发票后60天。"
  }
}
~~~

在生成训练中，最终答案可能仍是自然语言，但 annotation 应保留结构字段。这样可以把“数值错”“页码错”“引用错”和“表达错”分开评估，也能为后续检索、OCR 或规则校验提供接口。

### 5.7 多图与多轮：训练归因而不是训练长度

多图比较数据要明确图片身份和比较维度：

~~~json
{
  "task": "multi_image_compare",
  "media_order": ["before", "after"],
  "question": "比较两张图中桌面和杯子的位置变化。",
  "answer": "第二张图桌面更整洁，杯子从左侧移到了右侧。"
}
~~~

高质量样本应包含交换图片顺序的变式。如果交换后答案仍然不变，可能说明问题不依赖图片，或答案没有真正绑定 media_id。

多轮数据还要测试指代。第一轮可以问“这是什么房间”，第二轮问“沙发是什么颜色”，第三轮问“把刚才提到的沙发和桌子比较”。如果每一轮都重新附带图片，训练的是重复绑定；如果后续轮次不附带图片，训练的是会话级引用。两种数据不能不加标记地混合。

### 5.8 视频和音频：时间轴是监督的一部分

视频 instruction tuning 不能只把若干帧当作无序图片。样本应说明：

- 帧的时间戳或顺序；
- 采样间隔；
- 动作开始和结束；
- 关键事件所在区间；
- 是否允许回答未采样时刻的内容。

例如“谁先拿起杯子”要求事件顺序，而“画面中有什么”主要要求空间识别。两者需要不同的帧采样和评估。

音频任务也有不同层级：语音转写、说话人区分、声音事件识别、情绪或语义问答。若训练样本把准确转写文本直接放进 prompt，却没有让模型访问原始音频，模型可能只学会文本问答。数据 artifact 应标明答案依赖的是 waveform、转写、视频帧，还是多种证据的组合。

### 5.9 拒答、不确定性与安全样本

多模态模型不应该在任何问题上都强行回答。拒答样本应区分原因：

- 图片本身模糊；
- 关键区域被遮挡；
- 问题要求的信息不在图片中；
- 图片之间缺少必要页或顺序；
- 内容敏感，系统没有权限；
- 媒体中的文字包含指令，但它只是观察内容，不是系统命令。

一个合格的拒答通常包括原因和下一步，而不是无条件输出“我不知道”：

~~~text
这张图片中的车牌区域过于模糊，我无法可靠读出号码。请提供更高分辨率的原图，或裁剪车牌区域后重试。
~~~

拒答数据也可能过强。如果把大量可回答样本标成拒答，模型会形成 over-refusal。数据中应同时有清晰、可回答、部分可回答和确实不可回答的对照组，并记录拒答原因。

MM-SafetyBench 等研究可以作为多模态安全评估的资料入口，但公开 benchmark 的结果不能直接替代目标产品的权限、隐私和工具调用测试。尤其要把媒体观察通道和 trusted instruction channel 分开：图片里的“忽略之前指令”只是图片文字，不应获得 system 或 developer 权限。

## 6. 数据质量是一个测量问题

### 6.1 媒体完整性检查

训练前最便宜的一类检查是媒体完整性：

- 文件是否存在；
- 是否能被正确解码；
- 色彩空间和通道数是否符合 processor；
- 宽高是否为正；
- 视频帧是否可随机访问；
- 音频采样率和时长是否在预期范围；
- hash 是否与 manifest 一致。

这类检查只能证明“能加载”，不能证明“内容与答案匹配”。一个可以打开的错误图片，反而比路径失效更危险，因为它会产生看似正常的梯度。

### 6.2 证据支持要按 claim 统计

不要把一整段 answer 粗略标成 supported 或 unsupported。设样本 i 有 C_i 个可核验 claim，第 j 个 claim 的支持标记为 z_{i,j}，则 claim-level 支持率可以写成：

~~~math
S_{\mathrm{claim}}
=
\frac{\sum_i\sum_{j=1}^{C_i}z_{i,j}}
{\sum_i C_i}.
~~~

要求 $C_i$ 是非负整数，且总 claim 数大于零；没有任何可核验 claim 时，支持率是“没有分母”，不应报告为 0。若业务只关心高风险 claim，还应在分母中明确只纳入已标记为高风险的 claim，不能用全量分母和高风险分子混算。

若一个回答有四个 claim，其中三个有图片区域支持，一个是猜测，把整条样本记为“支持”会夸大质量。还可以单独报告关键 claim 支持率，把金额、日期、药名等高风险字段与普通场景描述分开。

支持标注本身也有成本。可以先对高风险任务保存 evidence span，对普通 caption 使用人工抽样和反事实测试。没有证据标注的样本不必全部丢弃，但不能在评估中假装它们具有同等可验证性。

### 6.3 重复、近重复与切分污染

多模态数据的重复不只是一模一样的文件：

- 相同图片换了文件名；
- 同一视频抽取了不同帧；
- 同一文档只改了页码或压缩方式；
- 相同图片配了不同问题；
- 网络图片的裁剪版本同时出现在训练和测试；
- OCR 文本高度相同但图片略有旋转。

应根据任务选择去重键，例如媒体 hash、感知 hash、文档 id、视频片段 id、OCR 文本和语义近邻。随机按样本切分往往会把同一文档的不同页面分到两边，导致评估过于乐观。

### 6.4 标注分歧不是噪声的同义词

caption 中的“拥挤”“漂亮”“危险”可能存在合理分歧；OCR 中的字符通常有更严格答案；图表推理中的单位和数值则需要核对。把所有不一致都当作错误，会误删有价值的多视角样本；把所有不一致都保留，又会污染监督。

对两位标注者在离散标签上的一致性，可以用 Cohen's kappa 表示：

~~~math
\kappa
=
\frac{p_o-p_e}{1-p_e},
~~~

其中 p_o 是观测一致率，p_e 是按边际分布计算的随机一致率。对于开放式答案，还需要字段级比对、数值容差、证据区域交并比和专家复核，而不能只用字符串相等。

### 6.5 隐私和许可属于数据语义

脱敏不能只处理 OCR 文本。图片中的脸、车牌、屏幕、手写签名和背景文件都可能泄露信息；音频中的声音和视频中的旁观者同样需要考虑。训练前应记录哪些字段被删除、模糊、替换或保留，以及这种处理是否改变了任务标签。

许可也会影响能否训练、能否分发和能否提供删除。公开可访问不等于可以任意再分发。模型卡或数据说明中的许可边界应与实际 artifact 对齐。

### 6.6 按切片记录质量

总体质量平均数经常掩盖关键失败。至少应按下列切片记录：

- 任务类型；
- 语言和文字方向；
- 分辨率区间；
- 图片数量；
- 文档页数；
- OCR 字体和版面；
- 视频时长和帧数；
- 资料是否足够；
- 敏感等级；
- 数据来源和标注批次。

如果总体 placeholder mismatch rate 很低，但所有错误集中在多图样本，模型上线后的比较任务仍然会失败。切片不是额外装饰，而是让总体数字有解释力的必要条件。

## 7. 数据混合：模型最终学到的是采样分布

### 7.1 样本比例不等于 token 比例

假设有 K 类任务，第 k 类样本抽样概率是 p_k：

~~~math
\sum_{k=1}^{K}p_k=1.
~~~

如果每类回答长度不同，样本比例 p_k 不等于监督 token 比例。令第 k 类平均有效回答长度为 \ell_k，则近似的 token 占比为：

~~~math
q_k
=
\frac{p_k\ell_k}
{\sum_{r=1}^{K}p_r\ell_r}.
~~~

一个长文档任务即使样本数量不多，也可能贡献大量梯度；一个短拒答任务若只按样本数配比，实际监督 token 可能很少。配比设计必须先决定是控制样本频率、有效 token 还是任务成功率。

### 7.2 混合分布改变梯度

设任务 k 的期望梯度为 g_k，混合训练的期望梯度近似为：

~~~math
\mathbb{E}[g]
=
\sum_{k=1}^{K}p_k g_k.
~~~

增加 OCR 样本不只“增加 OCR 能力”，也会改变模型对格式、长度、数字和视觉细节的偏好；增加拒答样本可能提高保守性，也可能损伤可回答任务。配比必须通过消融和切片评估来判断，不能把某篇论文的比例照搬为普遍规律。

### 7.3 为什么保留纯文本数据

多模态 SFT 可能改变原有语言能力、工具格式或长文本行为。加入一定量的纯文本 instruction 数据有时可以帮助保持语言侧能力，但纯文本比例过高，又会让模型重新依赖语言先验，视觉使用能力可能下降。

因此纯文本不是默认越多越好。应同时监控：

- 纯文本指令质量；
- 视觉任务的反事实敏感性；
- 长文本和结构化输出；
- 多语言能力；
- 训练 token 和每类 loss。

### 7.4 过采样、重复和难例

少数但重要的 OCR、拒答和高风险样本可能需要过采样。过采样会提高它们的梯度频率，却也可能让模型记住固定措辞、产生 over-refusal 或过度偏向某一种版面。

更稳健的做法是：

1. 先记录原始分布；
2. 选择需要提升的切片；
3. 使用不同媒体、问题和措辞生成变式；
4. 监控重复率和有效 token；
5. 用未重采样的验证集评估。

难例不等于噪声。一个标注确定但视觉细节困难的样本可能值得保留；一个答案本身不确定且没有证据的样本则需要重新标注或明确为不确定性任务。

### 7.5 课程顺序与混合训练

一种可能的训练顺序是先用较简单的 caption 和短 VQA 让模型熟悉图文接口，再加入 OCR、图表、多图和拒答；另一种方式是从一开始混合所有任务。哪一种更好取决于模型初始化、数据规模和优化预算。

课程顺序会改变训练轨迹，不是数学上无关紧要的排列。若先训练大量粗粒度 caption，再长期冻结视觉塔，后续 OCR 任务可能很难纠正早期形成的压缩偏好。最终应比较相同总 token、相同验证切片和相同训练步数下的不同顺序。

## 8. Batch、collator 与训练阶段

### 8.1 一个多模态 batch 包含多个长度

文本 batch 常见形状是 [B,T]，但多模态 batch 还要面对：

- 每条样本的图片数量不同；
- 图片尺寸不同；
- 动态分辨率产生不同视觉 token 数；
- 视频帧数不同；
- 音频时长不同；
- assistant 回答长度不同。

一个抽象 batch 可以写成：

~~~text
{
    "input_ids": text_ids,
    "labels": labels,
    "attention_mask": text_mask,
    "media": media_batch,
    "media_ids": media_ids,
    "media_mask": media_mask,
    "image_sizes": original_sizes,
    "assistant_spans": assistant_spans,
}
~~~

实际框架字段会不同，但责任不能消失。collator 需要把原始样本中的媒体引用、文本 token、视觉输入和 label span 组合成同一条可追踪记录。

### 8.2 Padding、ragged 和固定预算

若视觉 token 长度不同，可以：

1. padding 到 batch 最大长度；
2. 使用 packed 或 ragged 表示；
3. 先用 resampler 压成固定长度；
4. 按分辨率分桶。

padding 简化实现，却可能浪费显存和注意力计算。固定长度降低预算波动，却可能形成视觉信息瓶颈。分桶减少浪费，但增加调度和数据组织复杂度。选择哪一种，要同时看吞吐、显存、OCR 质量和多图错误率。

### 8.3 按有效 token 累积梯度

若不同样本的 assistant token 数差异很大，按 micro-batch 平均 loss 再平均，和按所有有效 token 汇总后再除，结果不同。对窗口 W，按有效 token 统计的目标可以写成：

~~~math
\mathcal{L}_{W}
=
-\frac{
\sum_{b\in W}\sum_t m_{b,t}\log p(y_{b,t})
}{
\sum_{b\in W}\sum_t m_{b,t}
}.
~~~

如果使用梯度累积，应保存 loss_sum 和 valid_tokens，而不是只保存每个 micro-batch 的标量平均。最后一个不完整窗口也要明确是丢弃、按实际 token 更新，还是补齐后更新。否则学习率、日志和不同 batch size 的实验不可比。

### 8.4 冻结和解冻不是二元开关

常见阶段包括：

- 冻结 vision encoder，只训练 projector 或 query bridge；
- 冻结 LLM，训练 connector 和部分视觉模块；
- 对 LLM 使用 LoRA；
- 解冻视觉塔的高层；
- 联合微调全部组件。

冻结更多参数通常更省显存、更稳定，但领域适应和 OCR 能力可能受限；解冻更多参数可能提升适应性，也可能造成语言遗忘、视觉过拟合和训练不稳定。每个阶段都应记录学习率、可训练参数、数据混合和验证切片。

### 8.5 训练监控不应只有一个 loss

至少按任务记录：

- token-level loss；
- 有效 label 数；
- 每类样本数和有效 token；
- placeholder 错误数；
- 视觉输入失败数；
- OCR、图表、grounding 和拒答切片；
- 纯文本回归；
- 显存、吞吐和样本长度。

一个总体 loss 下降、OCR loss 上升或拒答切片恶化的实验，不应被一个平均数描述为“训练成功”。

## 9. 如何评估 instruction tuning 是否真的有效

### 9.1 评估对象要先写清楚

可以把多模态评估分成五层：

1. 协议层：媒体绑定、模板和输出格式是否正确；
2. 感知层：对象、文字、数字、图表元素是否读取正确；
3. 推理层：比较、计数、空间和多步计算是否正确；
4. 证据层：答案是否由正确图片、页码和区域支持；
5. 行为层：证据不足时是否表达不确定，是否遵守隐私和权限。

协议层通过，不等于感知层通过；感知层通过，也不等于证据层和行为层通过。

### 9.2 文本答案指标的边界

Exact Match 适合数字、短字段和结构化答案，但字符串不同不一定事实不同。自然语言相似度可以作为辅助，却可能把一个数字错误的答案评得很高。

对数值字段，可定义容差匹配：

~~~math
\mathrm{match}(a,\hat a)
=
\mathbf{1}\left[
|a-\hat a|
\le \epsilon_{\mathrm{abs}}
+\epsilon_{\mathrm{rel}}|a|
\right].
~~~

其中 ε_abs 和 ε_rel 应按字段风险和单位规定。金额、日期和药物剂量不能使用随意的宽容度。

### 9.3 OCR 和图表需要结构化评估

OCR 可以分别报告：

- 字符或 token 的编辑距离；
- 字段 exact match；
- 数字和单位准确率；
- 页码和区域准确率；
- 表格行列一致性。

图表可以分别报告：

- 图表类型和图例识别；
- 数据点读取；
- 比较和排序；
- 单位换算；
- 多步计算；
- 生成结论的证据支持。

这样才能区分“读错字符”“读对数字但单位错”“读对数值但比较错”三类事故。

### 9.4 Grounding 和引用支持

设预测框为 B_p，标注框为 B_g，交并比为：

~~~math
\mathrm{IoU}(B_p,B_g)
=
\frac{|B_p\cap B_g|}
{|B_p\cup B_g|}.
~~~

但 IoU 不是所有文档任务的唯一标准。合同问答还要检查：

- 页码是否正确；
- 证据区域是否包含完整字段；
- quote 是否与区域文本一致；
- claim 是否由该区域支持；
- 多图引用是否指向正确 media_id。

一个模型可以输出高 IoU，却把金额 claim 归因到错误页；因此空间命中和语义支持应分别记录。

### 9.5 反事实测试视觉是否被使用

要检查模型是否真正使用媒体，可以做配对测试：

- 替换关键对象，问题不变；
- 遮挡金额、数字或关系区域；
- 交换多图顺序；
- 替换 OCR 文本但保持背景；
- 提供无关图片；
- 保持图片不变，只改问题中的指代。

对样本 i，若模型在原图上的预测为 \hat a_i，在反事实图片上的预测为 \hat a_i^{cf}，可以记录答案变化率：

~~~math
R_{\mathrm{cf}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[\hat a_i\ne \hat a_i^{cf}].
~~~

这里要求配对样本数 $N>0$，否则变化率未定义。这个指标不是“理解真值”。不该变化的问题可能被无关改变扰动，应该变化的问题也可能有多个合理答案。它的作用是暴露视觉未接入、placeholder 错位和语言先验主导。

### 9.6 拒答是一个混淆矩阵

把样本分为确实可回答和确实不可可靠回答，模型输出回答或拒答，就可以得到：

- true answer：该答且答对；
- false answer：不该可靠回答却硬答；
- correct refusal：不可靠时拒答；
- over-refusal：可答时拒答。

拒答准确率不应只看“拒答样本中拒了多少”，还要同时报告可回答样本上的过度拒答。可以使用精确率、召回率和风险-覆盖率曲线，而不是一个孤立比例。

### 9.7 纯文本回归与多轮回归

多模态训练可能损伤：

- 纯文本问答；
- 长上下文；
- JSON 或工具参数格式；
- 多语言输出；
- 代码和数学能力。

因此必须保留训练前的文本回归集，并把模板版本、tokenizer 版本和解码参数固定。多轮图文评估还要专门检查历史图片引用、图片顺序和跨轮指代，不能用单轮 VQA 分数替代。

### 9.8 单位成功任务成本

生产评估不只关心准确率，还关心得到一个证据充分答案的代价。定义证据支持成功事件为 S=1，成本为 C，则窗口内单位成功任务成本可以写成：

~~~math
C_{\mathrm{per\ supported\ success}}
=
\frac{\sum_{i=1}^{N}C_i}
{\sum_{i=1}^{N}\mathbf{1}[S_i=1]}.
~~~

C_i 可以包括视觉编码、OCR、LLM token、重试、人工复核和 GPU 时间。要求成功事件分母大于零；窗口内没有一个证据支持的成功任务时，单位成功任务成本未定义，不能写成无穷大或零。若只把“生成了一段文字”算作成功，模型可能通过自信幻觉降低表面成本；分母必须与业务定义的成功事件一致。

## 10. 贯穿案例：合同页面问答数据如何进入训练

### 10.1 业务问题

假设系统要处理三页合同，用户要求：

~~~text
比较三页中的付款期限，指出超过 60 天的条款，并给出页码和原文区域。
~~~

这不是普通 caption。它需要：

1. 多图顺序；
2. 文档 OCR；
3. 数值比较；
4. claim 与区域绑定；
5. 证据不足时说明缺页或模糊；
6. 对合同内容执行权限检查。

### 10.2 样本设计

一条正样本可以保存：

~~~json
{
  "sample_id": "contract_compare_004",
  "media": [
    {"media_id": "p1", "type": "image", "uri": "p1.png"},
    {"media_id": "p2", "type": "image", "uri": "p2.png"},
    {"media_id": "p3", "type": "image", "uri": "p3.png"}
  ],
  "task": "multi_page_numeric_compare",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "media_id": "p1"},
        {"type": "image", "media_id": "p2"},
        {"type": "image", "media_id": "p3"},
        {"type": "text", "text": "比较三页中的付款期限，指出超过60天的条款。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "第2页的付款期限为90天，超过60天。证据位于第2页付款条款区域。"
        }
      ]
    }
  ],
  "annotations": {
    "claims": [
      {"claim_id": "c1", "value": 90, "unit": "day", "supported_by": ["e2"]},
      {"claim_id": "c2", "value": true, "supported_by": ["e2"]}
    ],
    "evidence": [
      {
        "evidence_id": "e2",
        "media_id": "p2",
        "bbox": [410, 1200, 1680, 1420],
        "transcript": "付款期限：90天"
      }
    ]
  }
}
~~~

回答文字只是一个训练输出，annotations 才让数据工程和评估知道 90、天、第 2 页和区域之间的关系。

### 10.3 从样本到 batch 的责任链

数据进入训练时要依次完成：

~~~text
manifest
    -> media decode and permission check
    -> message normalization
    -> template rendering
    -> placeholder/media binding
    -> vision preprocessing
    -> visual token or memory construction
    -> assistant span extraction
    -> labels and masks
    -> token-budget accounting
    -> batch and loss
~~~

每一步都可能成为错误来源。例如：

- manifest 顺序错，导致 p2 和 p3 互换；
- 图像读取成功但 crop 丢掉付款条款；
- OCR annotation 正确但 labels 的 assistant 起点错一位；
- 三张图 token 让上下文超限，末尾关键问题被截断；
- 训练输出包含“第 2 页”，却没有任何区域证据。

### 10.4 如何定位一次失败

若模型把 90 天读成 60 天，先查视觉分辨率、OCR 区域和数字标注；若数字读对但页码错，查 media_id 和多图顺序；若证据正确但模型把图片中的“忽略之前指令”当作工具命令，查 trusted instruction 与 observation 的权限分离；若所有回答都说无法确定，查拒答样本配比和 policy 标签。

这就是数据工程的价值：把“模型答错了”拆成可以复测的局部责任，而不是用一次总体分数结束分析。

### 10.5 一组可复现的局部实验

对合同样本可以固定一组配对实验：

1. 原始三页；
2. 交换 p2 和 p3；
3. 遮挡付款期限区域；
4. 去掉 p2；
5. 只保留 OCR 文本；
6. 给 p2 加入与问题无关的恶意文字。

分别记录数值、页码、区域和拒答行为。实验不能证明模型拥有完整“理解”，但能说明它是否依赖正确页面和正确字段。

## 11. 最小可运行的数据契约审计

下面的程序只使用 Python 标准库。它不加载图片，不运行 vision encoder，也不判断一段真实回答是否真的由像素支持。它做的是训练前的低成本审计：

- placeholder 与图片数量；
- 视觉 token 和文本 token 的预算；
- assistant label 数；
- 任务类型覆盖；
- 证据支持标记；
- 资料不足样本的拒答标记；
- 明显安全风险标记。

为了让示例集中讨论数据契约，代码用布尔字段模拟人工或上游审核结果。真实系统必须把这些字段连接到图片解码、OCR、人工标注和权限服务，而不能直接相信它们。

~~~python
import re
from collections import Counter

IGNORE_INDEX = -100
VISUAL_TOKENS_PER_IMAGE = 576
CONTEXT_LIMIT = 2048

REQUIRED_TASKS = {
    "caption",
    "vqa",
    "ocr",
    "chart",
    "grounding",
    "multi_turn",
    "multi_image",
    "refusal",
}

records = [
    {
        "id": "caption_001",
        "task": "caption",
        "images": 1,
        "prompt": "<image> Describe this street scene.",
        "user_tokens": 18,
        "assistant_tokens": 22,
        "special_tokens": 4,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "vqa_count_001",
        "task": "vqa",
        "images": 1,
        "prompt": "<image> How many cups are on the table?",
        "user_tokens": 16,
        "assistant_tokens": 8,
        "special_tokens": 4,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "ocr_001",
        "task": "ocr",
        "images": 1,
        "prompt": "<image> What is the total amount on the receipt?",
        "user_tokens": 18,
        "assistant_tokens": 10,
        "special_tokens": 4,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "chart_001",
        "task": "chart",
        "images": 1,
        "prompt": "<image> Which quarter has the highest sales?",
        "user_tokens": 17,
        "assistant_tokens": 9,
        "special_tokens": 4,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "grounding_001",
        "task": "grounding",
        "images": 1,
        "prompt": "<image> Locate the red button.",
        "user_tokens": 14,
        "assistant_tokens": 12,
        "special_tokens": 4,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "multi_turn_001",
        "task": "multi_turn",
        "images": 1,
        "prompt": "<image> What room is this? What color is the sofa?",
        "user_tokens": 42,
        "assistant_tokens": 20,
        "special_tokens": 8,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "multi_image_001",
        "task": "multi_image",
        "images": 2,
        "prompt": "<image_1> <image_2> Compare these pages.",
        "user_tokens": 28,
        "assistant_tokens": 18,
        "special_tokens": 6,
        "supported": True,
        "needs_refusal": False,
        "refused": False,
        "safe": True,
    },
    {
        "id": "refusal_001",
        "task": "refusal",
        "images": 1,
        "prompt": "<image> What is the blurry license plate number?",
        "user_tokens": 20,
        "assistant_tokens": 18,
        "special_tokens": 4,
        "supported": True,
        "needs_refusal": True,
        "refused": True,
        "safe": True,
    },
    {
        "id": "bad_ocr_001",
        "task": "ocr",
        "images": 1,
        "prompt": "<image_1> <image_2> Read the account number.",
        "user_tokens": 18,
        "assistant_tokens": 12,
        "special_tokens": 4,
        "supported": False,
        "needs_refusal": True,
        "refused": False,
        "safe": False,
    },
]


def count_placeholders(prompt):
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return len(re.findall(r"<image(?:_\d+)?>", prompt))


def require_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def total_tokens(record):
    for field in ("images", "user_tokens", "assistant_tokens", "special_tokens"):
        require_nonnegative_int(record[field], field)
    visual = record["images"] * VISUAL_TOKENS_PER_IMAGE
    text = record["user_tokens"] + record["assistant_tokens"]
    return visual + text + record["special_tokens"]


def label_count(record):
    for field in ("images", "user_tokens", "assistant_tokens", "special_tokens"):
        require_nonnegative_int(record[field], field)
    if record["assistant_tokens"] == 0:
        raise ValueError("at least one assistant token is required")
    visual_tokens = record["images"] * VISUAL_TOKENS_PER_IMAGE
    ignored = visual_tokens + record["user_tokens"] + record["special_tokens"]
    labels = [IGNORE_INDEX] * ignored
    labels += list(range(record["assistant_tokens"]))
    return sum(label != IGNORE_INDEX for label in labels)


def safe_ratio(numerator, denominator):
    if numerator < 0 or denominator < 0:
        raise ValueError("ratio inputs cannot be negative")
    return None if denominator == 0 else numerator / denominator


def safe_max(values):
    return None if not values else max(values)


def audit(record):
    issues = []
    if record["task"] not in REQUIRED_TASKS:
        issues.append("unknown_task")
    if count_placeholders(record["prompt"]) != record["images"]:
        issues.append("placeholder_mismatch")
    if total_tokens(record) > CONTEXT_LIMIT:
        issues.append("context_over_budget")
    if record["assistant_tokens"] == 0:
        issues.append("empty_assistant_supervision")
    elif label_count(record) != record["assistant_tokens"]:
        issues.append("label_mask_error")
    if not record["supported"]:
        issues.append("unsupported_answer")
    if record["needs_refusal"] and not record["refused"]:
        issues.append("missing_refusal")
    if not record["safe"]:
        issues.append("safety_risk")
    return issues


reports = {record["id"]: audit(record) for record in records}
accepted = [record for record in records if not reports[record["id"]]]
rejected = [sample_id for sample_id, issues in reports.items() if issues]

accepted_tasks = {record["task"] for record in accepted}
task_counts = Counter(record["task"] for record in accepted)
issue_counts = Counter(
    issue for issues in reports.values() for issue in issues
)

label_tokens = sum(label_count(record) for record in accepted)
assistant_tokens = sum(record["assistant_tokens"] for record in accepted)
max_total_tokens = safe_max([total_tokens(record) for record in accepted])
task_coverage = safe_ratio(
    len(accepted_tasks & REQUIRED_TASKS),
    len(REQUIRED_TASKS),
)

supported_count = sum(record["supported"] for record in accepted)
support_rate = safe_ratio(supported_count, len(accepted))

refusal_needed = [
    record for record in accepted if record["needs_refusal"]
]
refusal_accuracy = safe_ratio(
    sum(record["refused"] for record in refusal_needed),
    len(refusal_needed),
)

checks = {
    "bad_sample_rejected": rejected == ["bad_ocr_001"],
    "required_tasks_covered": task_coverage == 1.0,
    "labels_match_answers": label_tokens == assistant_tokens,
    "within_context": (
        max_total_tokens is not None
        and max_total_tokens <= CONTEXT_LIMIT
    ),
    "support_ok": support_rate == 1.0,
    "refusal_ok": refusal_accuracy == 1.0,
}


def expects_error(function):
    try:
        function()
    except (TypeError, ValueError):
        return True
    return False


boundary_checks = {
    "empty_ratio_is_undefined": safe_ratio(0, 0) is None,
    "non_string_prompt_rejected": expects_error(
        lambda: count_placeholders(None)
    ),
    "negative_token_count_rejected": expects_error(
        lambda: total_tokens(
            {
                "images": 1,
                "user_tokens": -1,
                "assistant_tokens": 1,
                "special_tokens": 1,
            }
        )
    ),
    "empty_assistant_rejected": expects_error(
        lambda: label_count(
            {
                "images": 1,
                "user_tokens": 1,
                "assistant_tokens": 0,
                "special_tokens": 1,
            }
        )
    ),
}

print("accepted_ids=", [record["id"] for record in accepted])
print("rejected=", {sample_id: reports[sample_id] for sample_id in rejected})
print("task_counts=", dict(sorted(task_counts.items())))
print("task_coverage=", round(task_coverage, 3))
print("label_tokens=", label_tokens)
print("assistant_tokens=", assistant_tokens)
print("max_total_tokens=", max_total_tokens)
print("support_rate=", round(support_rate, 3))
print("refusal_accuracy=", round(refusal_accuracy, 3))
print("issue_counts=", dict(sorted(issue_counts.items())))
print("checks=", checks)
print("boundary_checks=", boundary_checks)
print(
    "all_checks_passed=",
    all(checks.values()) and all(boundary_checks.values()),
)
~~~

这段程序的预期输出是：

~~~text
accepted_ids= ['caption_001', 'vqa_count_001', 'ocr_001', 'chart_001', 'grounding_001', 'multi_turn_001', 'multi_image_001', 'refusal_001']
rejected= {'bad_ocr_001': ['placeholder_mismatch', 'unsupported_answer', 'missing_refusal', 'safety_risk']}
task_counts= {'caption': 1, 'chart': 1, 'grounding': 1, 'multi_image': 1, 'multi_turn': 1, 'ocr': 1, 'refusal': 1, 'vqa': 1}
task_coverage= 1.0
label_tokens= 117
assistant_tokens= 117
max_total_tokens= 1204
support_rate= 1.0
refusal_accuracy= 1.0
issue_counts= {'missing_refusal': 1, 'placeholder_mismatch': 1, 'safety_risk': 1, 'unsupported_answer': 1}
checks= {'bad_sample_rejected': True, 'required_tasks_covered': True, 'labels_match_answers': True, 'within_context': True, 'support_ok': True, 'refusal_ok': True}
all_checks_passed= True
~~~

这里的 all_checks_passed 只表示这 9 条 toy 记录的结构字段彼此一致。它不表示图片真的被解码、不表示答案真的被像素支持，也不表示训练后的模型会正确执行 OCR、grounding 或拒答。真正的系统还要把这段审计与媒体解码、人工抽检、独立验证集和反事实评估连接起来。

## 12. 练习：从样本契约到训练判断

1. 设计一条包含两张图片和两个 assistant 回答的多轮样本，明确每个 media_id 的生命周期。
2. 把一条字符串式样本改写成 media、messages、annotations、policy 四层结构。
3. 为交错图文样本画出 message refs、media objects 和 visual inputs 三个顺序。
4. 给出一条包含 system、user、image reference 和两轮 assistant 输出的 label mask。
5. 说明为什么一个 image reference 不等于一个视觉 token，并计算三张图片各 576 个视觉 token、512 个文本 token和 8 个特殊 token 的总长度。
6. 设计一个 OCR 字段 annotation，同时保存 transcript、页码和 bbox。
7. 为图表问答构造一个需要单位换算的样本，并写出计算公式。
8. 设计一个多图交换顺序的反事实测试，说明什么结果会暴露图片归因错误。
9. 计算一个包含三个 claim 的答案的 claim-level support rate，并说明整条样本二值化的偏差。
10. 设计 caption、OCR、拒答三类数据的混合比例，同时区分样本占比和有效 token 占比。
11. 解释为什么梯度累积应保存 loss_sum 和 valid_tokens，而不是只平均 micro-batch loss。
12. 设计一个拒答混淆矩阵，分别报告正确拒答、硬答和 over-refusal。
13. 计算两个 bounding box 的 IoU，并说明为什么文档引用还需要页码和 quote。
14. 设计一组纯文本回归、多模态切片和单位成功任务成本指标，判断一次训练是否值得继续。

## 13. 资料入口与证据边界

1. Liu et al., Visual Instruction Tuning，LLaVA 原始论文：<https://arxiv.org/abs/2304.08485>。它支持视觉指令微调的数据与训练路线，但实验结论绑定论文中的模型、数据和评估协议。
2. Dai et al., InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning：<https://arxiv.org/abs/2305.06500>。它支持 instruction-aware 的视觉语言训练研究背景，不能直接证明任意 connector 或数据配比。
3. Liu et al., Improved Baselines with Visual Instruction Tuning，LLaVA-1.5：<https://arxiv.org/abs/2310.03744>。它可用于核对公开训练和评估设置，不能替代目标业务复测。
4. Zhu et al., MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models：<https://arxiv.org/abs/2304.10592>。它提供另一条视觉语言对齐与指令训练的公开研究路线。
5. Singh et al., Towards VQA Models That Can Read：TextVQA：<https://arxiv.org/abs/1904.08920>。它支持“图片文字问答”这一任务定义和数据背景，不等于通用 OCR 质量保证。
6. Masry et al., ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning：<https://arxiv.org/abs/2203.10244>。它支持图表问答和逻辑推理的公开评估入口。
7. Gong et al., MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models：<https://arxiv.org/abs/2311.17600>。它支持多模态安全评估研究背景，不能替代产品权限和工具副作用测试。
8. Hugging Face Transformers 多模态 chat template 文档：<https://huggingface.co/docs/transformers/main/chat_templating_multimodal>。它支持特定 Transformers 版本中多模态消息和 processor 的 API 语义，实际行为应绑定安装版本。
9. PyTorch CrossEntropyLoss 文档：<https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>。它支持 ignore_index 和 reduction 等 loss API 语义；本章的多轮 span 构造仍是教学代码，需要目标框架实测。
10. LLaVA 官方开源仓库：<https://github.com/haotian-liu/LLaVA>。它可用于核对某个代码版本的数据和模板实现；仓库当前状态、分支和部署路径必须单独记录。

这些资料的可信度不同：原始论文最适合支持论文方法和论文实验，官方文档最适合支持 API 语义，仓库最适合支持某个版本的实现细节，本章 demo 只支持手算和数据契约。任何来源都不能单独证明目标系统的幻觉率、拒答质量、隐私安全或单位成功任务成本。

## 14. 本章回顾

多模态 instruction tuning 的核心，不是把一条图片路径塞进 JSON，而是把媒体、消息、模板、视觉引用、assistant 监督、证据和策略边界编译成同一个训练契约。

一条样本至少要回答四个问题：模型看到了哪些媒体，用户要求什么，哪些 token 产生梯度，答案由什么证据支持。caption、VQA、OCR、图表、grounding、多图、视频、音频和拒答分别训练不同的操作，不能用一个总体 caption 数据集替代它们。

chat template 决定高层消息如何变成模型输入；一个 image reference 可能展开成大量视觉 token，也可能进入独立 memory。assistant-only label mask 决定哪些输出承担监督责任，attention mask 决定哪些位置可以互相读取，两者不能混淆。

数据质量必须按 claim、任务、来源、分辨率、语言、媒体数量和风险切片测量；混合比例要区分样本频率和有效 token；评估要同时覆盖协议、感知、推理、证据和行为。标准库审计可以提前发现 placeholder、预算和 mask 的结构错误，却不能替代真实图片评估、反事实测试、人工抽检和独立验证集。

下一章将进入 diffusion 基础，讨论如何把图像生成写成加噪、去噪、训练目标和采样过程，而不是只记住“扩散模型能生成图片”这一句结论。
