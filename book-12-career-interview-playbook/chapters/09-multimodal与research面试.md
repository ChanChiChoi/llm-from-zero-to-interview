# 第九章：Multimodal 与 Research 面试

Multimodal 与 Research 面试考的不是你能不能背出 CLIP、LLaVA、Diffusion、Whisper、Sora、DPO、FlashAttention 这些名字，而是你能不能把一个新方向放进正确问题里，讲清它解决什么瓶颈、方法为什么有效、实验是否可信、边界在哪里，以及如果让你复现或改进，你会怎么做。

很多候选人在这一关会出现两个问题。第一个问题是多模态知识碎片化：知道 CLIP 做图文对齐，但讲不清 contrastive learning；知道 VLM 能看图，但讲不清 vision encoder、connector 和 LLM 如何连接；知道 diffusion 能生成图像，但讲不清加噪、去噪和条件控制；知道视频生成难，但只会说“算力大”。第二个问题是研究表达空泛：能报论文名，但不会判断贡献；能说“可以改进”，但没有实验设计；能说“这个方法很好”，但不讲 baseline、ablation、失败模式和成本。

这一章把多模态面试和研究讨论面试放在一起，是因为它们经常同时出现。多模态方向变化快，面试官更看重你的研究判断力，而不是只看你是否背过固定答案。

本章重点：CLIP、VLM、Diffusion、视频生成、语音模型、多模态评估、安全、论文讨论、复现实验、ablation、开放研究题。

## 9.1 Multimodal 与 Research 面试到底考什么

这类面试通常围绕两组能力。

第一组是多模态技术能力：

1. 图文对齐：CLIP、contrastive learning、image-text retrieval。
2. 视觉编码：ViT、CNN、patch embedding、视觉特征层级。
3. VLM 架构：vision encoder、connector、LLM、image token、multimodal instruction tuning。
4. 生成模型：diffusion、latent diffusion、text-to-image、text-to-video。
5. 语音音频：ASR、TTS、speech encoder、端到端语音对话。
6. 多模态评估：OCR、图表、空间关系、计数、视觉幻觉、安全。

第二组是研究讨论能力：

1. 如何介绍一篇论文。
2. 如何判断贡献是否真实。
3. 如何复现核心结论。
4. 如何设计 ablation。
5. 如何解释负结果。
6. 如何提出下一步研究方向。
7. 如何处理开放问题和不确定问题。

这些能力可以用一句话串起来：多模态面试考你是否理解不同模态如何表示、对齐、融合、生成和评估；研究面试考你是否能用科学方法判断一个新方法是否真的有效。

## 9.2 回答多模态题的通用结构

多模态题建议用“五步结构”：

1. 先说任务：图文检索、图片问答、图像生成、视频生成、语音识别还是多模态 Agent。
2. 再说输入输出：输入有哪些模态，输出是什么，监督信号是什么。
3. 讲核心架构：encoder、connector、LLM、decoder、diffusion U-Net 或 DiT。
4. 讲训练目标：contrastive loss、caption loss、next-token loss、denoising loss、instruction tuning。
5. 讲评估和失败模式：幻觉、OCR、计数、空间关系、时序一致性、安全和成本。

例如问“VLM 如何把图像接入 LLM”，不要只说：

```text
用 vision encoder 提取图像特征，然后喂给 LLM。
```

更好的回答是：

```text
一个典型 VLM 会先用 vision encoder，比如 ViT，把图像切成 patch 并编码成视觉 token；然后用 connector，比如 linear projection、MLP 或 Q-Former，把视觉特征映射到 LLM 的 hidden space；最后把这些 image tokens 和文本 tokens 一起送入 LLM，让 LLM 以自回归方式生成答案。训练上通常先做图文对齐或 caption 预训练，再做多模态 instruction tuning。关键问题是视觉特征是否保留足够细节、image token 数量和成本如何权衡、训练数据是否覆盖 OCR、图表、空间关系和多轮问答。
```

这个回答有架构、训练、成本和失败模式。

## 9.3 回答研究讨论题的通用结构

研究讨论题建议用“七步论文模板”：

1. 背景：这篇论文之前大家怎么做。
2. 问题：前人方法遇到什么瓶颈。
3. 方法：论文提出了什么核心机制。
4. 实验：数据、baseline、指标和 ablation 是否支撑结论。
5. 边界：适用条件、失败模式、隐藏成本。
6. 复现：如果你做，最小复现路径是什么。
7. 下一步：如何改进或迁移到新场景。

如果面试官问“介绍一篇你熟悉的多模态论文”，可以这样组织：

```text
我会先说明它解决的具体问题，比如图文对齐、VLM instruction tuning 或视频生成一致性。然后讲之前方法为什么不够，再讲它的核心机制。接着看实验是否和强 baseline、公平数据和合理指标比较，是否有关键 ablation。最后我会讲它的边界，比如数据成本、推理成本、评估偏差、安全风险，以及如果我要复现，会先用小模型和公开数据验证哪个核心结论。
```

## 9.4 CLIP 如何训练

CLIP 的核心目标是把图像和文本映射到同一个语义空间，使匹配的 image-text pair 更接近，不匹配的 pair 更远。

典型结构包括：

1. Image encoder：通常是 ResNet 或 ViT。
2. Text encoder：通常是 Transformer。
3. Projection head：把图像和文本表示映射到同一维度。
4. Contrastive loss：在一个 batch 内让正确图文对相似度最高。

面试回答：

```text
CLIP 使用图文对比学习。一个 batch 里有 N 对 image-text pair，模型分别编码图像和文本，得到归一化 embedding，然后计算 N*N 的相似度矩阵。训练目标是让第 i 张图像和第 i 段文本相似度最高，同时把 batch 中其他文本作为负样本；反过来文本到图像也做同样目标。这样模型学到跨模态语义对齐，可以用于 zero-shot classification 和图文检索。
```

常见追问：“CLIP 为什么能 zero-shot 分类？”

回答：

```text
因为 CLIP 训练时已经把图像和自然语言描述对齐到同一空间。做分类时，可以把类别名写成 prompt，例如 a photo of a dog，把每个类别 prompt 编码成文本 embedding，再和图像 embedding 比相似度。它不是训练一个固定分类头，而是利用文本空间定义类别。
```

CLIP 的局限：

1. 依赖大规模图文对，数据噪声和偏见会进入模型。
2. 对细粒度计数、空间关系、OCR 不一定强。
3. 相似度对齐不等于深层推理。
4. prompt wording 会影响 zero-shot 结果。
5. 训练目标偏检索和对齐，不直接生成长回答。

## 9.5 Vision Encoder 面试怎么答

Vision encoder 的作用是把图像变成可供后续模型使用的视觉表示。

常见问题包括：

1. CNN 和 ViT 的区别。
2. Patch embedding 是什么。
3. 为什么 VLM 常用预训练 vision encoder。
4. 图像分辨率如何影响能力和成本。
5. OCR、细粒度识别和空间关系为什么难。

面试回答：

```text
Vision encoder 负责把原始像素转成视觉 token 或全局 embedding。CNN 带有局部卷积和归纳偏置，适合提取局部纹理；ViT 把图像切成 patch，用 self-attention 建模 patch 间关系，更容易和 Transformer 语言模型生态对接。VLM 中常用预训练 vision encoder，是因为从零训练视觉表示成本高，CLIP 或类似模型已经学到较好的图文语义对齐。
```

图像分辨率的 trade-off：

1. 分辨率越高，细节、OCR、小物体更容易保留。
2. 分辨率越高，patch token 数越多，计算和显存更高。
3. 下采样过强会丢失文字、表格和细粒度空间信息。
4. 多尺度或 crop 策略可以缓解，但系统复杂度更高。

## 9.6 VLM 架构：图像如何接入 LLM

VLM 常见架构可以拆成三块：

1. Vision encoder：提取视觉特征。
2. Connector：把视觉特征映射到 LLM hidden space。
3. LLM：把 image tokens 和 text tokens 作为上下文生成回答。

Connector 常见形式：

1. Linear projection：简单、成本低。
2. MLP projector：表达更强。
3. Q-Former 或 resampler：用少量 query 压缩视觉 token。
4. Cross-attention：让语言模型某些层 attend 视觉特征。

面试回答：

```text
VLM 的核心问题是视觉表示和语言模型表示空间不一致。Connector 负责把 vision encoder 输出的视觉 token 映射到 LLM 能理解的 hidden space。简单 projector 成本低，但可能保留细节不足；Q-Former 或 resampler 可以压缩大量视觉 token，降低上下文成本，但可能损失细粒度信息。架构选择本质是在视觉细节、推理能力、训练成本和推理成本之间权衡。
```

常见追问：“为什么 VLM 会视觉幻觉？”

回答：

```text
视觉幻觉可能来自多个环节：vision encoder 没捕捉到细节，connector 压缩丢信息，LLM 语言先验太强，训练数据里图文不一致，或者指令微调奖励了流畅回答而不是基于图像证据。评估时要区分看错、没看见、语言补全和拒绝承认不确定几类错误。
```

## 9.7 多模态 Instruction Tuning

多模态 instruction tuning 的目标是让模型学会按照用户指令理解图像、文本和对话上下文。

数据形式通常是：

1. 单图问答。
2. 多轮图文对话。
3. OCR 和文档理解。
4. 图表解释。
5. 空间关系、计数和定位。
6. 安全拒答和不确定性表达。

训练中要注意：

1. Image placeholder 和视觉 token 对齐。
2. Chat template 一致。
3. 通常只对 assistant response 算 loss。
4. 图片和文本样本长度差异大，要控制 packing 和 padding。
5. 数据质量比数量更重要，错误 caption 会强化幻觉。

面试回答：

```text
多模态 instruction tuning 不只是把图片和问答拼起来。关键是让模型学会在图像证据约束下回答用户指令。训练时要保证 image token、chat template 和 assistant-only loss mask 一致；数据要覆盖描述、OCR、图表、计数、空间关系、多轮对话和拒答边界；评估时要看视觉 grounding，而不是只看语言流畅度。
```

## 9.8 Diffusion Model 如何生成图像

Diffusion model 的核心思想是学习从噪声逐步还原数据。

训练过程：

1. 从真实图像开始逐步加噪。
2. 模型学习在某个时间步预测噪声或去噪方向。
3. 推理时从随机噪声开始，多步去噪得到图像。
4. 文本条件通过 text encoder 和 cross-attention 或 conditioning 注入。

面试回答：

```text
Diffusion 可以理解为学习一个反向去噪过程。训练时把真实图像加不同强度的噪声，让模型预测加入的噪声或干净图像；推理时从纯噪声出发，按时间步逐步去噪，最终得到图像。文本到图像模型会把 prompt 编码成条件，通过 cross-attention 或其他条件机制影响去噪过程。
```

常见追问：“Latent Diffusion 为什么高效？”

回答：

```text
Latent Diffusion 不直接在像素空间做扩散，而是先用 autoencoder 把图像压到 latent space，在低分辨率潜空间里去噪，再解码回图像。这样显著降低计算和显存成本，同时尽量保留语义和视觉质量。代价是 autoencoder 压缩可能损失细节。
```

Diffusion 的优缺点：

1. 优点：生成质量高、训练稳定、条件控制灵活。
2. 缺点：采样步数多、推理慢、精确文本和空间控制困难。
3. 风险：偏见、版权、深伪、隐私和安全滥用。

## 9.9 视频生成为什么难

视频生成比图像生成更难，因为视频同时要求空间质量和时间一致性。

难点包括：

1. 数据量大：视频 token 或 latent 序列远大于图像。
2. 时序一致：人物、物体、背景不能每帧漂移。
3. 物理合理：运动、接触、遮挡、因果关系要可信。
4. 长程规划：视频要有持续事件结构。
5. 评估困难：自动指标很难衡量真实观感和世界一致性。
6. 成本高：训练、存储、采样和人工评估都贵。

面试回答：

```text
视频生成不是简单地逐帧生成图像。逐帧生成会导致闪烁、身份漂移和运动不连续。好的视频模型需要同时建模空间细节和时间动态，还要理解动作、物理约束、镜头变化和长程事件结构。训练上需要高质量视频文本数据，系统上要处理巨大计算成本，评估上也不能只看单帧质量。
```

常见追问：“视频生成和 world model 有什么关系？”

回答：

```text
视频生成需要预测未来视觉状态，因此和 world model 有相似目标：都要建模环境动态、物体持久性和动作后果。但当前视频生成模型生成逼真视频，不等于拥有可交互、可规划、可验证的完整世界模型。是否具备 world model 能力，需要看它能否支持反事实预测、动作条件控制、长期一致性和下游决策。
```

## 9.10 Whisper 与语音模型

语音方向常见问题包括 ASR、TTS、语音对话和音频理解。

ASR 的目标是把语音转成文本。典型系统包括：

1. Audio feature extraction，例如 log-mel spectrogram。
2. Encoder，把音频特征编码成序列表示。
3. Decoder 或 CTC，把表示转成文本 token。
4. 多语言和时间戳处理。

面试回答：

```text
语音识别和文本 LLM 的区别在于输入是连续音频信号，需要先转成时频特征或 audio tokens，再建模时间序列。Whisper 这类模型通过大规模弱监督语音文本数据学习多语言 ASR、翻译和时间戳能力。部署时要关注噪声、口音、说话人重叠、延迟和下游 LLM 对 ASR 错误的鲁棒性。
```

语音接入 LLM 的两种路线：

1. Pipeline：ASR 转文本，再给 LLM，最后 TTS。
2. End-to-end：语音 encoder 直接接入语言模型或统一模型。

Pipeline 简单可控，端到端潜力更大但训练和评估更复杂。

## 9.11 多模态评估怎么设计

多模态评估不能只问“图片里有什么”。

应覆盖：

1. Caption：整体描述是否准确。
2. OCR：文字识别是否正确。
3. Chart/Table：图表和表格理解。
4. Counting：数量判断。
5. Spatial relation：位置、方向、遮挡。
6. Fine-grained recognition：细粒度类别和属性。
7. Multi-image reasoning：多图比较和变化检测。
8. Visual grounding：回答是否基于图像证据。
9. Safety：危险图像、隐私、人脸、版权、深伪。

面试回答：

```text
我会把多模态评估拆成感知、理解、推理、生成和安全几层。感知层看 OCR、计数、细粒度识别；理解层看图表、文档、场景关系；推理层看多图、多证据和空间关系；生成层看描述质量和幻觉；安全层看隐私、敏感内容和不当建议。只用通用 VQA 分数无法覆盖真实产品风险。
```

多模态评估的风险：

1. Benchmark 污染。
2. 题目过于简单。
3. 只测最终答案，不测证据定位。
4. LLM judge 对图像不可见或偏差大。
5. 不评估延迟、成本和失败类型。

## 9.12 多模态安全

多模态模型引入新的安全面。

常见风险：

1. 图像中的隐私信息：人脸、证件、地址、屏幕截图。
2. OCR prompt injection：图片里的文字诱导模型忽略系统指令。
3. 视觉误识别导致错误建议。
4. 深伪、版权和生成内容滥用。
5. 医疗、法律、金融等高风险图像解释。
6. 多模态越狱：用图像承载文本或符号绕过安全策略。

面试回答：

```text
多模态安全要同时考虑视觉内容和文本指令。图片不是可信输入，OCR 出来的文字也可能包含 prompt injection。系统上要区分用户指令、图像内容和外部文档内容；高风险场景要限制能力或要求人工复核；评估上要覆盖隐私、OCR 注入、危险内容、视觉误识别和正常请求误拒。
```

## 9.13 如何介绍一篇多模态论文

如果面试官问“讲一篇你熟悉的多模态论文”，推荐模板：

```text
我会先讲它解决的任务，例如图文对齐、VLM 问答、图像生成或视频生成。然后讲前人方法的瓶颈，比如视觉特征和语言空间不对齐、图像 token 太多、生成时序不一致、评估不可靠。接着讲方法核心，包括架构、训练目标、数据和推理流程。然后看实验是否包含强 baseline、关键 ablation、成本分析和失败案例。最后讲边界，比如数据依赖、推理成本、幻觉、安全风险，以及我会如何最小复现。
```

不要只讲论文结论，要主动讲边界。

例如讲 LLaVA，可以强调：它的重要性在于用相对简单的 vision encoder + projector + LLM 加 instruction tuning 跑通多模态对话范式；边界是视觉细节、OCR、复杂图表、幻觉和评估可靠性。

例如讲 Stable Diffusion，可以强调：它把扩散过程放到 latent space 降低成本；边界是文本精确控制、手部和空间关系、版权和安全风险。

## 9.14 如何判断多模态论文贡献是否真实

判断多模态论文，要重点看六点：

1. 任务是否真实：是不是解决真实视觉语言问题，而不是只优化 demo。
2. Baseline 是否强：是否和已有强 VLM、diffusion 或 video model 比较。
3. 数据是否公平：是否使用更多更干净数据造成提升。
4. 指标是否可靠：是否只看主观样例或单一 benchmark。
5. 是否有 ablation：架构、数据、分辨率、token 数、训练阶段各自贡献。
6. 成本是否透明：训练成本、推理延迟、显存、人工标注和数据处理成本。

面试回答：

```text
我会特别警惕多模态论文中的 demo bias。漂亮样例不等于稳定能力。需要看公开 benchmark、真实业务样本、失败案例、消融和成本。如果论文同时换了更大模型、更多数据、更高分辨率和更复杂 prompt，就很难判断提升来自哪里。
```

## 9.15 如何复现一个 VLM 项目

最小复现路线可以这样设计：

1. 选择小规模开源 LLM 和预训练 vision encoder。
2. 用 linear projector 或 MLP connector 对齐视觉特征。
3. 准备小型 image-text instruction 数据。
4. 跑通单图问答和 caption。
5. 评估 OCR、计数、图表、空间关系和幻觉样本。
6. 做 ablation：冻结/解冻 vision encoder、不同 connector、不同 image token 数、不同数据类型。

面试回答：

```text
如果让我复现一个 VLM，我不会一开始追求大规模效果，而会先验证视觉特征能否通过 connector 被 LLM 使用。最小版本可以冻结 vision encoder 和 LLM，只训练 projector，然后在小型图文问答数据上看模型是否学会引用图像信息。之后再做 instruction tuning、OCR/图表专项评估和幻觉分析。
```

关键坑：

1. Image token 与文本模板不一致。
2. 只看语言流畅度，不看图像 grounding。
3. 数据中 caption 错误导致幻觉。
4. 训练样本太单一，模型只学到模板。
5. 评估集被训练数据污染。

## 9.16 如何设计多模态 ablation

多模态 ablation 要对应明确假设。

常见消融：

1. Vision encoder：CLIP ViT vs 其他 encoder。
2. Connector：linear、MLP、Q-Former、resampler。
3. Image token 数：更多 token 是否提升 OCR 和细节理解。
4. 分辨率：高分辨率对小文字和细节的收益。
5. 数据类型：caption、VQA、OCR、chart、multi-turn 的贡献。
6. 训练阶段：预对齐、instruction tuning、偏好优化各自贡献。
7. 冻结策略：冻结 vision encoder、冻结 LLM、全参微调或 LoRA。

好的回答：

```text
如果论文声称提升来自更好的 connector，我会固定 vision encoder、LLM、训练数据和分辨率，只替换 connector，并同时看总体 VQA、OCR、计数、图表和延迟成本。如果多个变量同时变了，就不能证明 connector 是核心贡献。
```

## 9.17 开放研究题怎么回答

开放研究题不是要你给出唯一正确答案，而是看你如何建模问题。

推荐结构：

1. 先复述问题和目标。
2. 拆出关键瓶颈。
3. 提出 2-3 个假设。
4. 为每个假设设计实验。
5. 说明指标、风险和 trade-off。
6. 给出优先级。

例如问“如何设计下一代多模态模型”，可以回答：

```text
我会从架构、数据、训练、评估和安全五个维度设计。架构上考虑更统一的多模态 token 表示，减少手工 connector；数据上需要高质量图文、视频、语音、文档和交互数据；训练上结合对齐预训练、instruction tuning 和偏好优化；评估上覆盖 OCR、图表、视频时序、工具使用和安全；系统上关注成本、延迟和权限。短期我会先做更可靠的评估和数据闭环，长期再探索统一架构。
```

## 9.18 高频研究题：最近哪篇论文重要

这类题的重点不是论文名，而是判断标准。

可以先说：

```text
我判断一篇论文重要，会看它是否抓住真实瓶颈，是否改变后续方法范式，是否有可信实验支持，是否能影响工程系统或研究路线。
```

然后选一篇你能讲深的论文。

如果选择 CLIP：强调图文对比学习和 zero-shot 迁移。

如果选择 Stable Diffusion：强调 latent diffusion 降低生成成本。

如果选择 LLaVA：强调用 instruction tuning 把 VLM 对话范式工程化。

如果选择 FlashAttention：强调 IO-aware 系统视角。

如果选择 DPO：强调偏好优化流程简化。

回答时必须补边界，不要只夸。

## 9.19 高频研究题：如何解释负结果

多模态和研究复现很容易出现负结果。

排查顺序：

1. 数据：图文是否匹配，caption 是否错误，训练集是否太小。
2. 模板：image placeholder、chat template、loss mask 是否正确。
3. 模型：vision encoder、connector、LLM hidden size 是否匹配。
4. 训练：learning rate、batch size、冻结策略、混合精度是否合理。
5. 评估：指标是否覆盖目标能力，样本是否被污染。
6. 规模：方法是否依赖更大模型或更多数据。

面试回答：

```text
负结果不一定说明方法无效。我会先确认实现和数据没有错误，再检查是否复现了论文关键条件。如果仍无效，会分析是规模不足、数据质量差、baseline 更强、指标不匹配，还是论文结论本身不稳。最后把排查过程整理成负结果报告，说明哪些假设被排除，下一步如何验证。
```

## 9.20 面试中的常见失分点

Multimodal 与 Research 面试常见失分点包括：

1. 把 VLM 说成“图像特征拼到 LLM”，不讲 connector 和训练目标。
2. 讲 CLIP 只说图文匹配，不讲 batch 内对比学习。
3. 讲 diffusion 只说“从噪声生成图”，不讲训练和反向去噪。
4. 讲视频生成只说“算力大”，不讲时序一致和物理约束。
5. 多模态评估只看 VQA，不看 OCR、图表、计数、幻觉和安全。
6. 研究讨论只背论文名，不讲问题、机制、实验和边界。
7. 判断论文只看 leaderboard，不看 baseline 和 ablation。
8. 提改进点太空泛，没有对应失败模式和实验。
9. 复现方案只说“跑开源代码”，不讲最小复现和控制变量。
10. 遇到不熟论文时假装看过，而不是给分析框架。

## 9.21 高频题回答模板

### 9.21.1 CLIP 如何训练

```text
CLIP 用图文对比学习训练 image encoder 和 text encoder。一个 batch 中有 N 对图文样本，模型计算图像和文本 embedding 的 N*N 相似度矩阵，让匹配的图文对相似度最高，不匹配的作为负样本。这样学到共享语义空间，可用于图文检索和 zero-shot 分类。
```

### 9.21.2 VLM 如何接入图像

```text
典型 VLM 用 vision encoder 把图像编码成视觉 token，再用 connector 把视觉特征映射到 LLM hidden space，最后把 image tokens 和 text tokens 一起送入 LLM 生成回答。训练上通常先做图文对齐，再做多模态 instruction tuning。关键 trade-off 是视觉细节、image token 数、训练数据和推理成本。
```

### 9.21.3 Diffusion 如何生成图像

```text
Diffusion 训练一个反向去噪过程。训练时给真实图像加不同程度噪声，让模型预测噪声或干净图像；推理时从随机噪声开始逐步去噪。文本到图像模型会把 prompt 编码成条件，通过 cross-attention 或其他 conditioning 控制生成内容。
```

### 9.21.4 视频生成为什么难

```text
视频生成不仅要单帧质量，还要时间一致性、物体持久性、运动合理性和长程事件结构。逐帧生成容易闪烁和身份漂移。训练数据、计算成本和评估难度也远高于图像生成，因此视频生成比图像生成复杂很多。
```

### 9.21.5 如何评价一篇论文贡献

```text
我会先看它解决的问题是否真实重要，再看前人方法为什么不够，然后判断新方法的核心机制和假设。实验上要看强 baseline、公平数据、关键 ablation、成本和失败案例。最后看边界和可复现性。如果只在弱 baseline 或小样例上提升，没有消融和成本分析，我会比较谨慎。
```

## 9.22 一套完整 Multimodal + Research 面试回答模板

如果被问开放题：“请你设计一个多模态助手，并说明如何研究和评估它”，可以这样组织：

```text
第一，我会先明确任务边界：它是图片问答、文档理解、图表分析、多图推理、语音对话，还是图文生成系统。不同任务需要不同架构和评估。

第二，设计模型架构。图片理解场景可以用 vision encoder、connector 和 LLM；生成场景可能需要 diffusion 或 video generation 模型；语音场景可以先用 ASR+LLM+TTS pipeline，再考虑端到端方案。

第三，设计训练数据。数据要覆盖 caption、VQA、OCR、chart、spatial relation、counting、多轮对话和安全边界，并保证 image token、chat template 和 loss mask 一致。

第四，设计评估。不能只看通用分数，要分层评估感知、理解、推理、grounding、幻觉、安全、延迟和成本。对产品场景还要做 bad case 分类。

第五，设计研究闭环。如果引入新 connector、新数据或新训练目标，要用 ablation 控制其他变量，比较强 baseline，并分析失败模式。负结果也要记录原因。

第六，设计上线风险。多模态输入可能包含隐私、OCR prompt injection、版权和危险内容，因此要有权限控制、内容安全、日志脱敏和人工复核策略。
```

## 9.23 准备清单

准备 Multimodal 与 Research 面试时，至少要能回答下面的问题：

1. CLIP 如何训练？为什么能 zero-shot 分类？
2. Vision encoder 在 VLM 中起什么作用？
3. VLM 如何把图像接入 LLM？
4. Connector 有哪些形式，各有什么 trade-off？
5. 多模态 instruction tuning 要注意什么？
6. 为什么 VLM 会视觉幻觉？
7. Diffusion 的训练和推理过程是什么？
8. Latent Diffusion 为什么更高效？
9. 视频生成为什么比图像生成难？
10. 语音模型和文本 LLM 的区别是什么？
11. 如何设计多模态评估集？
12. 多模态安全有哪些新风险？
13. 如何介绍一篇你熟悉的论文？
14. 如何判断论文贡献是否真实？
15. 如何设计 ablation？
16. 如何复现一个 VLM 或 DPO 项目？
17. 如何解释负结果？
18. 如何提出一个不空泛的研究改进点？
19. 如果现场遇到没读过的论文，如何分析？
20. 如何把多模态论文变成可展示项目？

这一章的核心不是让你背所有多模态模型名，而是建立一种研究型表达方式：先定义问题，再讲机制和证据，最后讲边界、实验和下一步。
