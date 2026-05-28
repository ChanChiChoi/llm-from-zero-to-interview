# 第六章：Training 面试

Training 面试考的不是你会不会说“预训练、SFT、ZeRO、Megatron、loss spike、checkpoint”这些词，而是你能不能像一个真正做过大规模训练的人一样，把训练目标、数据、模型、优化器、并行策略、稳定性、评估和成本串成一个闭环。

很多候选人准备 training 面试时容易走偏：背很多分布式训练名词，但讲不清为什么要做数据清洗；知道 AdamW，但讲不清学习率和 batch size 的关系；知道 loss spike，但不知道第一时间应该排查数据、数值、学习率、checkpoint 还是集群故障；知道 checkpoint，但不知道保存频率会影响故障恢复成本和训练吞吐。

大模型训练是一个系统工程。它不是“把数据喂给模型跑很久”这么简单，而是在有限预算下，稳定地把高质量 token 转化为模型能力。

本章重点：数据、预训练、分布式训练、loss spike、评估、checkpoint、成本估算。

## 6.1 Training 面试到底考什么

Training 面试通常围绕七条主线：

1. 训练目标：预训练、SFT、偏好学习分别在优化什么。
2. 数据工程：数据质量、去重、混合比例、采样策略如何影响模型能力。
3. 优化与稳定性：learning rate、batch size、warmup、梯度裁剪、loss spike、数值精度。
4. 分布式训练：data parallel、tensor parallel、pipeline parallel、ZeRO、FSDP、通信瓶颈。
5. 训练监控：loss、gradient norm、throughput、MFU、显存、eval 指标。
6. Checkpoint 与恢复：保存什么、多久保存、如何 resume、如何处理坏 checkpoint。
7. 成本估算：参数量、token 数、FLOPs、GPU 数、训练时长、预算。

这些主题看似分散，其实可以用一句话串起来：训练团队先定义目标能力和训练预算，再准备高质量数据，选择模型规模和训练配方，通过分布式系统高效训练，并用监控、评估和 checkpoint 保证训练过程稳定可恢复，最后根据结果迭代数据和配方。

面试官想看到的不是你背出某个框架的 API，而是你能不能回答这些问题：

1. 如果 loss 突然炸了，你怎么排查？
2. 如果训练很慢，你怎么判断瓶颈在计算、通信、I/O 还是 checkpoint？
3. 如果模型评估不好，你怎么判断是数据问题、模型规模问题、训练配方问题还是评估问题？
4. 如果预算有限，你会怎么估算能训练多大的模型、多少 token、跑多久？
5. 如果训练中断，你如何安全恢复并确认没有 silent bug？

## 6.2 回答 Training 题的通用结构

Training 面试建议使用“五步结构”：

1. 先定义目标：是在预训练、SFT、继续预训练，还是偏好学习阶段。
2. 再说输入输出：输入是什么数据，输出希望模型获得什么能力。
3. 解释核心机制：loss、优化器、并行方式、训练循环。
4. 讲工程边界：稳定性、吞吐、显存、通信、checkpoint、成本。
5. 给排障思路：指标异常时如何定位。

例如问“怎么做一次大模型预训练”，不要只说：

```text
准备数据，搭模型，用 AdamW 训练，最后评估。
```

更好的回答是：

```text
我会先确定目标模型规模、训练 token 数和预算，然后准备多来源高质量语料，做清洗、去重、质量过滤和数据混合。训练目标通常是 causal language modeling，用 next-token prediction 优化交叉熵。工程上需要选择并行策略，比如 data parallel、tensor parallel、pipeline parallel 或 ZeRO/FSDP，并监控 loss、learning rate、gradient norm、throughput、显存和 eval 指标。训练过程中要定期保存 checkpoint，处理节点故障和 loss spike。最后用通用 benchmark、领域评估、人工分析和下游任务验证能力，并根据结果回到数据和训练配方继续迭代。
```

这个回答体现了完整闭环。

## 6.3 预训练：模型从哪里获得能力

预训练是大模型能力的主要来源。它的目标通常是让模型在大规模 token 序列上学习语言、事实、代码、推理模式、格式和世界知识。

对于 decoder-only LLM，最常见目标是 causal language modeling：

```text
给定前面的 token，预测下一个 token。
```

训练 loss 通常是所有有效 token 上的负 log likelihood：

```text
Loss = - average log P(x_t | x_1, ..., x_{t-1})
```

面试时要强调：预训练不是显式教模型“回答问题”，而是通过预测下一个 token 让模型压缩训练分布中的规律。问答、代码、推理和指令遵循能力，一部分来自预训练数据中的模式，一部分来自后续 SFT 和偏好学习。

预训练阶段常见设计问题包括：

1. 模型规模：参数量、层数、hidden size、attention head 数、context length。
2. 数据规模：训练 token 数、数据来源、去重、质量过滤、混合比例。
3. 训练配方：optimizer、learning rate schedule、batch size、warmup、weight decay、gradient clipping。
4. 系统策略：并行方式、混合精度、激活重计算、checkpoint、容错。
5. 评估方式：训练 loss、validation loss、下游 benchmark、污染检查、人工分析。

如果面试官问“为什么预训练 loss 降低不一定代表模型更好”，可以回答：

```text
训练 loss 只衡量模型对训练分布的拟合程度。它下降说明模型更擅长预测训练样本中的 token，但不保证泛化到目标任务。数据重复、benchmark 污染、低质量数据占比、验证集不代表真实场景，都可能让 loss 看起来很好但模型能力一般。所以训练中要同时看 validation loss、分领域评估、任务 benchmark、人工样例和安全评估。
```

## 6.4 数据是训练的第一变量

大模型训练里，数据不是训练前的附属步骤，而是决定模型上限的核心变量。

同样的模型规模和训练预算，数据质量不同，结果可能完全不同。高质量数据可以提升知识密度、推理样式、代码能力和指令表达；低质量数据会引入重复、模板页、垃圾文本、错误事实、毒性内容、隐私风险和格式噪声。

Training 面试中关于数据，常见追问包括：

1. 为什么要去重？
2. 为什么要做质量过滤？
3. 数据混合比例怎么决定？
4. 代码、数学、网页、论文数据的作用分别是什么？
5. 如何避免 benchmark 污染？
6. 如果某个能力差，是不是只要加对应数据就行？

### 6.4.1 去重为什么重要

去重解决的不只是存储浪费问题。

重复数据会带来几个影响：

1. 降低有效 token 多样性：模型反复看到相同内容，浪费训练预算。
2. 增加记忆风险：模型更容易复述训练样本。
3. 污染评估：如果 benchmark 或相似题进入训练集，评估分数会虚高。
4. 扭曲数据分布：某些网站、模板或文档被过度采样。

可以这样回答：

```text
去重的核心目的是提升有效数据多样性和评估可信度，而不只是减少数据量。精确去重可以删除完全重复样本，近似去重可以删除高度相似内容。对 LLM 来说，去重还能降低记忆和 benchmark contamination 风险。但去重也不能过度，因为某些高频格式、代码模板或重要知识反复出现本身也反映真实分布。
```

### 6.4.2 质量过滤不是越严越好

质量过滤的目标是提高 token 的平均价值，而不是把所有“不像百科全书”的文本都删掉。

过松会保留垃圾、广告、乱码、模板页和低信息密度文本；过严会损失长尾知识、多语言数据、口语对话、代码注释、论坛问答和真实用户表达。

常见质量信号包括：

1. 文本长度、字符分布、乱码比例。
2. 重复 n-gram 比例。
3. 语言识别结果。
4. 毒性、成人、暴力、隐私等安全标签。
5. perplexity 或小模型质量评分。
6. 来源域名、文档类型、人工标注质量。

面试回答要体现 trade-off：

```text
质量过滤不是简单地越严格越好。预训练数据需要覆盖真实语言分布和长尾知识，如果只保留高质量百科和论文，模型可能表达单一、对口语和真实用户输入适应差。更合理的做法是分来源、分语言、分任务设定过滤规则，再通过消融实验看不同数据比例对验证集和下游能力的影响。
```

### 6.4.3 数据混合比例决定能力形状

数据混合比例决定模型“长成什么样”。

网页数据提供广覆盖，书籍提供长文本和叙事结构，论文提供专业知识，代码提供形式化结构和程序能力，数学数据提供推理样式，对话数据提供交互格式。

如果代码比例高，模型可能代码能力强，但自然语言风格受影响；如果英文比例过高，多语言能力会弱；如果短文本过多，长上下文能力可能不足；如果高质量推理数据少，模型可能只会流畅表达，不擅长复杂推理。

面试中可以这样说：

```text
数据 mixture 不是静态经验值，而是训练配方的一部分。通常会先根据目标能力设定初始比例，例如通用网页、书籍、代码、数学、论文、多语言数据，然后通过小规模 ablation 或中途 checkpoint 评估不同能力曲线。如果发现代码能力不足，可以提高代码数据或继续预训练；如果多语言能力不足，要检查多语言数据质量、tokenizer 压缩率和采样比例。
```

## 6.5 SFT 与预训练的区别

SFT，也就是 supervised fine-tuning，通常在预训练之后进行。它的目标不是重新学习世界知识，而是让模型学会按照指令格式完成任务。

预训练数据通常是自然文本，目标是预测下一个 token。SFT 数据通常是 instruction-response、multi-turn dialogue、tool-use trace 或专家示范，目标是让模型在给定指令时输出符合人类期望的回答。

预训练像是让模型读完整个图书馆，SFT 像是教模型如何参加面试、如何回答用户、如何遵守格式。

常见面试问题：SFT 会不会让模型学到新知识？

可以回答：

```text
SFT 可以注入一些新知识，但它更主要的作用是改变模型行为和输出格式。因为 SFT 数据量通常远小于预训练数据，且覆盖范围有限，如果希望系统性补充领域知识，继续预训练或领域预训练通常更合适；如果希望模型学会按指令回答、拒答、使用工具、遵循风格，则 SFT 更合适。
```

SFT 训练中要注意：

1. 数据质量比数量更重要。
2. 响应格式要一致，否则模型会学到混乱风格。
3. 多轮对话要正确 mask loss，避免把用户输入也当作需要生成的目标。
4. 要平衡 helpfulness、factuality、safety 和 verbosity。
5. SFT 可能带来能力遗忘，需要保留通用能力评估。

## 6.6 Optimizer、学习率和 batch size

训练面试经常从优化器开始追问。

大模型训练常用 AdamW。它相比 SGD 更适合稀疏、高维、非平稳的深度网络优化；相比 Adam，AdamW 把 weight decay 从梯度更新中解耦，通常更稳定。

但面试中不要只背“AdamW 好”。要讲清几个关键点：

1. 学习率决定每一步参数更新幅度。
2. warmup 用来避免训练早期梯度和动量估计不稳定。
3. decay schedule 用来在后期减小更新幅度，提高收敛质量。
4. batch size 影响梯度噪声、吞吐和泛化。
5. gradient clipping 用来限制异常梯度，减少 loss spike 风险。

如果问“为什么大模型训练要 warmup”，可以回答：

```text
训练刚开始时，模型参数随机，梯度分布不稳定，Adam 的一阶和二阶矩估计也还不可靠。如果一开始就用大学习率，参数可能被异常梯度推到很差区域，造成 loss spike 或发散。warmup 通过逐步增大学习率，让优化器统计量和激活分布先稳定下来。
```

如果问“batch size 越大越好吗”，可以回答：

```text
不是。更大的 global batch size 可以提高硬件利用率，降低梯度噪声，并支持更稳定的吞吐，但它也可能降低更新步数，影响泛化，且需要调整学习率。训练效果通常由 token 数、batch size、learning rate schedule 和模型规模共同决定。工程上还要考虑显存、通信成本和梯度累积带来的延迟。
```

## 6.7 分布式训练：为什么不能只用 Data Parallel

当模型和 batch 足够大时，单卡无法容纳模型、激活、优化器状态和 batch 数据，需要分布式训练。

常见并行方式包括：

1. Data Parallel：每张卡放一份完整模型，处理不同 batch，反向后同步梯度。
2. Tensor Parallel：把单层矩阵运算切到多张卡上。
3. Pipeline Parallel：把不同层放到不同设备，形成流水线。
4. ZeRO/FSDP：切分参数、梯度和优化器状态，减少每卡显存占用。
5. Sequence Parallel 或 Context Parallel：针对长上下文切分序列维度。

### 6.7.1 Data Parallel

Data Parallel 最直观：每张 GPU 复制一份模型，不同 GPU 处理不同样本，反向传播后做 all-reduce 同步梯度。

优点是简单、扩展性好；缺点是每张卡都要放完整模型和优化器状态。当模型很大时，显存不够。

面试回答：

```text
Data parallel 适合模型能放进单卡、主要想扩大 batch 和吞吐的场景。瓶颈通常是梯度同步通信。模型越大，all-reduce 成本越高；网络带宽差时，扩展效率会下降。
```

### 6.7.2 Tensor Parallel

Tensor Parallel 把单层中的大矩阵乘法切到多张卡上。例如一个巨大的 linear layer，可以按列或按行切分权重矩阵。

优点是能训练单卡放不下的层；缺点是层内通信频繁，对高速互联要求高。

面试回答：

```text
Tensor parallel 主要解决单层参数或激活太大、单卡放不下的问题。它通常需要高带宽低延迟互联，因为每层前向和反向都可能涉及 collective communication。因此 TP 更适合放在同一节点或高速互联域内。
```

### 6.7.3 Pipeline Parallel

Pipeline Parallel 把不同层放到不同设备。例如前几层在 GPU 0，中间层在 GPU 1，后几层在 GPU 2。

优点是能切分深层模型；缺点是有 pipeline bubble，需要 micro-batch 提高利用率，并且调度更复杂。

可以这样回答：

```text
Pipeline parallel 适合模型层数很多、按层切分比较自然的场景。它的核心问题是流水线空泡，设备不能一直满负载。通常会用 micro-batch 把一个 global batch 拆成多个小批次，让不同 stage 同时工作。
```

### 6.7.4 ZeRO 与 FSDP

ZeRO 和 FSDP 的核心思想是：不要让每张卡都保存完整训练状态。

训练时显存主要来自四类：

1. 参数。
2. 梯度。
3. 优化器状态。
4. 激活。

对于 AdamW，优化器状态通常包括一阶矩和二阶矩，显存占用很大。ZeRO/FSDP 通过切分参数、梯度、优化器状态，显著降低每卡显存压力。

面试回答：

```text
ZeRO/FSDP 主要解决 data parallel 下每张卡重复保存参数、梯度和优化器状态导致的显存浪费。不同 stage 切分不同训练状态。代价是训练时需要 gather 或 reduce-scatter 参数和梯度，因此通信复杂度增加。选择时要权衡显存节省和通信开销。
```

## 6.8 混合精度、数值稳定性与 loss spike

大模型训练通常使用混合精度，比如 FP16、BF16 或 FP8。混合精度可以降低显存占用、提高吞吐，但会带来数值稳定性问题。

BF16 相比 FP16 指数范围更大，通常更稳定，因此很多大模型训练偏好 BF16。FP16 可能需要 loss scaling 来避免 underflow。FP8 更节省但对训练配方和硬件支持要求更高。

### 6.8.1 什么是 loss spike

Loss spike 指训练过程中 loss 突然异常升高，有时会恢复，有时会导致训练发散。

它可能来自：

1. 数据异常：坏样本、超长样本、乱码、重复异常、错误 mask。
2. 学习率过大：尤其是 warmup 不足或 schedule 设计不当。
3. 梯度异常：gradient norm 突增、梯度裁剪缺失。
4. 数值问题：FP16 overflow/underflow、NaN、Inf。
5. 模型结构问题：初始化、归一化、激活函数、残差尺度。
6. 分布式问题：某个 rank 数据不同步、通信错误、checkpoint 恢复不一致。
7. 代码 bug：loss mask、position id、attention mask、label shift 错误。

### 6.8.2 loss spike 怎么排查

面试中如果被问“训练时 loss spike 怎么办”，推荐按优先级回答：

```text
我会先判断 spike 是单点异常、持续发散，还是所有 rank 同时出现。然后检查最近 batch 的数据质量、长度分布、loss mask、是否有 NaN/Inf。接着看 learning rate、gradient norm、梯度裁剪和 optimizer state 是否异常。如果是 resume 后出现，要检查 checkpoint 是否完整、optimizer 和 scheduler 状态是否恢复一致。如果只在部分 rank 出现，要看数据分片、通信和硬件错误。处理上可以先回滚到最近稳定 checkpoint，降低学习率或加强 gradient clipping，再定位根因，而不是盲目继续训练。
```

这个回答有几个关键点：

1. 先分类：偶发、持续、全局、局部。
2. 再排数据：坏样本和 mask bug 很常见。
3. 再排优化：LR、gradient norm、clip。
4. 再排数值：NaN、Inf、精度格式。
5. 再排系统：rank、通信、checkpoint、硬件。
6. 最后恢复：回滚 checkpoint，谨慎重启。

### 6.8.3 不要只说“调小学习率”

很多候选人回答 loss spike 时只说“调小 learning rate”。这不够。

学习率确实可能是原因，但不是唯一原因。如果 loss spike 来自数据坏样本、label shift、attention mask bug、checkpoint 损坏或某个 rank 同步错误，调小学习率只是掩盖问题。

更好的表达是：

```text
调小学习率是一种缓解手段，但我会先定位 spike 的类型和触发条件。如果 spike 总是在某类数据出现，优先看数据和 mask；如果发生在 warmup 或 decay 边界，优先看 schedule；如果伴随 NaN/Inf，优先看混合精度和梯度；如果发生在恢复训练之后，优先看 checkpoint 和 optimizer state。
```

## 6.9 训练监控：哪些指标必须看

训练不是启动一个 job 然后等结果。大规模训练必须持续监控。

核心指标包括：

1. Train loss：训练集上的 next-token loss。
2. Validation loss：验证集 loss，最好按领域拆分。
3. Learning rate：确认 schedule 是否符合预期。
4. Gradient norm：监控梯度爆炸、异常 batch。
5. Weight norm 或 update norm：看参数更新是否异常。
6. Throughput：tokens/sec 或 samples/sec。
7. MFU：Model FLOPs Utilization，衡量硬件有效利用率。
8. GPU 显存：判断是否接近 OOM。
9. GPU utilization：判断是否计算瓶颈或等待瓶颈。
10. Communication time：判断分布式通信瓶颈。
11. Data loading time：判断 I/O 和 dataloader 是否拖慢。
12. Checkpoint time：判断保存是否影响训练效率。
13. NaN/Inf count：捕捉数值异常。

面试时可以强调：单一指标不可靠。

例如：

1. loss 正常但 throughput 下降，可能是 I/O 或通信瓶颈。
2. throughput 正常但 validation loss 变差，可能是过拟合或数据分布问题。
3. train loss 降得快但 benchmark 不涨，可能是训练数据和评估目标不匹配。
4. GPU utilization 低，可能是 dataloader、通信、checkpoint 或 pipeline bubble。

## 6.10 Evaluation：训练中如何判断模型真的变好了

大模型训练不能只看最终 loss。

评估通常分为四类：

1. 语言建模指标：train loss、validation loss、perplexity。
2. 通用能力 benchmark：知识、推理、数学、代码、多语言。
3. 领域评估：目标业务或研究方向的数据集。
4. 人工分析：真实 prompt、失败案例、安全边界、输出风格。

Training 面试中要注意一个点：评估本身也可能有问题。

常见评估风险包括：

1. Benchmark contamination：评估题进入训练集。
2. Prompt 格式不一致：不同模板导致分数不可比。
3. Decoding 参数不一致：temperature、top-p、max tokens 影响结果。
4. 样本量太小：结果波动大。
5. 只看平均分：掩盖某些能力退化。
6. 数据语言和领域偏置：不能代表真实用户。

如果问“validation loss 降了，但 benchmark 没涨，怎么办”，可以回答：

```text
我会先确认 benchmark 评估流程是否稳定，包括 prompt、解码参数、样本量和污染检查。然后看 validation set 是否代表目标能力。如果 loss 降低主要来自网页或模板数据，可能不转化为推理、代码或数学能力。接着按领域拆分 validation loss，并分析失败样例，判断是数据 mixture、模型容量、上下文长度、训练步数还是 SFT/评估格式问题。
```

## 6.11 Checkpoint：训练工程的安全绳

Checkpoint 是大规模训练的安全绳。训练越贵，checkpoint 越重要。

一个完整 checkpoint 通常包括：

1. 模型参数。
2. 优化器状态。
3. 学习率 scheduler 状态。
4. random seed 和 RNG state。
5. dataloader 或 sampler 状态。
6. 当前 step、token 数、训练配置。
7. 分布式切分信息。

只保存模型参数，通常不足以无缝 resume 训练。因为 optimizer state、scheduler 和数据采样位置不一致，可能导致训练轨迹改变。

### 6.11.1 checkpoint 保存频率怎么定

保存太频繁，会占用 I/O、存储和训练时间；保存太少，故障后损失太大。

需要权衡：

1. 集群故障率。
2. 单次 checkpoint 写入耗时。
3. 存储成本。
4. 可接受的重算 token 数。
5. 训练总预算。

可以这样回答：

```text
checkpoint 频率取决于故障率、保存开销和可接受的重算成本。如果集群不稳定或训练成本很高，需要更频繁保存；如果 checkpoint 很慢，会显著影响吞吐，则要优化异步保存、分片保存或降低频率。通常还会保留最近若干 checkpoint，以及关键里程碑 checkpoint，避免最后一个 checkpoint 损坏导致无法恢复。
```

### 6.11.2 resume 后要检查什么

恢复训练后不能只看 job 跑起来了。

要检查：

1. loss 是否与恢复前连续。
2. learning rate 是否正确。
3. optimizer step 是否正确。
4. gradient norm 是否异常。
5. 数据是否重复或跳过。
6. 分布式 rank 和 shard 是否一致。
7. eval 指标是否没有突然漂移。

面试回答：

```text
resume 后我会重点看 loss 曲线是否连续、LR 和 step 是否对齐、optimizer state 是否恢复、数据 sampler 是否从正确位置继续，以及是否出现 NaN/Inf 或 gradient norm 突变。大规模训练里，silent bug 比直接 crash 更危险，因为它可能浪费大量算力但直到后期才暴露。
```

## 6.12 成本估算：训练到底要多少钱

Training 面试非常喜欢问成本估算，因为它能区分“只懂概念”和“懂工程”的候选人。

训练成本可以从几个变量估算：

1. 参数量 N。
2. 训练 token 数 D。
3. 每 token 训练 FLOPs。
4. GPU 峰值算力。
5. 硬件利用率。
6. GPU 数量。
7. 单价和训练时长。

对 dense decoder-only Transformer，一个常用粗略估算是：训练 FLOPs 约等于：

```text
6 * 参数量 * 训练 token 数
```

这只是粗略估算，不包括所有 attention、embedding、通信、checkpoint、数据加载和低利用率损耗，但面试中足够用于数量级判断。

例如，一个 7B 模型训练 1T tokens：

```text
FLOPs ≈ 6 * 7e9 * 1e12 = 4.2e22
```

如果 GPU 理论算力是每张 300 TFLOPs，实际利用率 40%，有效算力为：

```text
300e12 * 0.4 = 1.2e14 FLOPs/s
```

如果使用 1024 张 GPU，总有效算力约为：

```text
1024 * 1.2e14 = 1.23e17 FLOPs/s
```

训练时间约为：

```text
4.2e22 / 1.23e17 ≈ 341000 seconds ≈ 95 hours ≈ 4 days
```

这个估算非常粗，但能说明思路。

面试时要补一句边界：

```text
这个公式是数量级估算，实际训练时间还会受 sequence length、attention 实现、并行策略、通信、checkpoint、数据加载、故障重启和 MFU 影响。所以真实排期通常要根据小规模 benchmark 或已有集群 profiling 修正。
```

## 6.13 Scaling Law：模型、数据和算力怎么权衡

Scaling law 的核心启发是：模型效果会随模型规模、数据规模和计算量呈规律性改善，但三者需要匹配。

如果模型很大但数据不足，会 undertrain，训练预算没有充分利用；如果数据很多但模型太小，模型容量可能不足；如果算力有限，需要在模型参数量和训练 token 数之间做 trade-off。

面试中可以这样讲：

```text
训练不是模型越大越好，也不是 token 越多越好，而是在给定 compute budget 下选择合适的模型规模和数据规模。Chinchilla 之后大家更重视 compute-optimal training，也就是在相同算力下，用相对更小但训练 token 更多的模型，可能比参数更大但训练不足的模型效果更好。
```

还要注意：scaling law 是趋势，不是万能公式。

它不直接解决：

1. 数据质量差异。
2. 推理、代码、数学等具体能力差异。
3. SFT 和偏好学习后的行为变化。
4. 评估污染和 benchmark 偏差。
5. 长上下文、多模态、工具使用等特殊能力。

## 6.14 高频题：如何设计一次继续预训练

继续预训练常用于领域适配，例如医学、法律、金融、代码、数学或企业内部知识。

面试官可能问：如果已有一个 base model，如何继续预训练提升某领域能力？

可以按这个结构回答：

```text
我会先明确目标领域和评估集，确认是知识缺失、术语不熟、风格不对，还是任务格式问题。然后准备领域数据，做清洗、去重、隐私和质量过滤，并和一定比例通用数据混合，避免 catastrophic forgetting。训练时使用较小学习率和合适 token budget，监控领域 validation loss 和通用能力 benchmark。训练后比较 base model、continued pretraining model 和 SFT model 的表现，分析是否真的提升目标能力，以及是否损伤通用能力或安全性。
```

关键点：

1. 继续预训练适合补领域分布和知识。
2. SFT 更适合补指令格式和回答风格。
3. 要混入通用数据防止遗忘。
4. 要有领域 eval 和通用 eval。
5. 要控制学习率，避免破坏已有能力。

## 6.15 高频题：训练慢怎么排查

如果面试官问“训练吞吐很低怎么办”，不要直接说“加 GPU”。

应该先定位瓶颈。

排查路径：

1. 看 GPU utilization：GPU 是否在忙。
2. 看 tokens/sec：实际吞吐是否低于预期。
3. 看 MFU：计算利用率是否低。
4. 看 dataloader time：数据加载是否慢。
5. 看 communication time：all-reduce、all-gather 是否占比高。
6. 看 pipeline bubble：pipeline stage 是否等待。
7. 看 checkpoint time：是否频繁阻塞训练。
8. 看 sequence packing：短样本 padding 是否浪费。
9. 看 kernel：attention、MLP、norm 是否使用高效实现。

可以这样回答：

```text
我会先用 profiling 把 step time 拆开，看时间花在 forward、backward、optimizer、communication、dataloader 还是 checkpoint。如果 GPU utilization 低但 dataloader time 高，优先优化数据读取和预处理；如果通信占比高，检查并行策略、bucket size、网络拓扑和 overlap；如果 padding 浪费严重，优化 packing；如果 checkpoint 阻塞，考虑异步或分片保存。只有定位瓶颈后，增加 GPU 才有意义，否则可能越扩越低效。
```

## 6.16 高频题：如何判断训练是否发散

训练发散不只是 loss 变大。可以从多个信号判断：

1. loss 持续上升或出现不可恢复 spike。
2. gradient norm 持续异常增大。
3. 出现 NaN 或 Inf。
4. 参数 norm 或 update norm 异常。
5. validation loss 同步恶化。
6. 生成样例变得重复、乱码或失去格式。

回答时可以说：

```text
我不会只看单个 step 的 loss，而会看趋势和相关指标。如果 loss 单点 spike 后恢复，可能是异常 batch；如果 loss、gradient norm、NaN/Inf 和 validation loss 同时异常，就更像训练发散。处理上先暂停或回滚到稳定 checkpoint，保存现场日志和 batch 信息，再排查数据、学习率、混合精度、梯度裁剪、代码改动和分布式状态。
```

## 6.17 高频题：预训练、SFT、RLHF 的训练差异

面试官可能让你比较不同训练阶段。

可以用表格回答：

| 阶段 | 数据 | 目标 | 主要作用 | 风险 |
| --- | --- | --- | --- | --- |
| 预训练 | 大规模自然文本、代码、论文等 | next-token prediction | 学语言、知识、模式和基础能力 | 数据质量、成本、污染、稳定性 |
| 继续预训练 | 领域或新分布数据 | next-token prediction | 补领域能力和分布适配 | 遗忘、过拟合、领域偏置 |
| SFT | 指令-回答、对话、专家示范 | 模仿高质量回答 | 学会听指令和输出格式 | 数据风格单一、幻觉、遗忘 |
| 偏好学习 | 人类或模型偏好数据 | 优化偏好目标 | 提升有用性、安全性和偏好一致性 | reward hacking、过度拒答、能力退化 |

然后补一句：

```text
这些阶段不是互相替代的。预训练提供基础能力，继续预训练适配领域，SFT 教模型如何回答，偏好学习进一步塑造行为。不同阶段的 loss 和 eval 不能简单横向比较，因为它们优化目标不同。
```

## 6.18 面试中的常见失分点

Training 面试常见失分点包括：

1. 只背框架名，不讲为什么需要这种并行方式。
2. 把 loss 当成唯一指标，不讨论 eval 和数据分布。
3. loss spike 只会说调小学习率。
4. 不知道 checkpoint 需要保存 optimizer、scheduler 和 sampler 状态。
5. 不会估算训练成本，只会说“需要很多 GPU”。
6. 把 SFT 当成万能知识注入手段。
7. 忽略数据质量、去重和污染检查。
8. 不知道训练吞吐低要先 profiling。
9. 混淆 DP、TP、PP、ZeRO/FSDP 的作用。
10. 不会讲 trade-off，只给绝对答案。

## 6.19 一套完整 Training 面试回答模板

如果被问一个开放题：“请你设计一次 7B 模型预训练”，可以这样组织：

```text
第一，我会先确定目标和预算，包括模型规模、context length、目标能力、训练 token 数、GPU 资源和时间约束。

第二，准备数据。数据会来自网页、书籍、代码、论文、数学、多语言等来源，经过清洗、去重、质量过滤、安全过滤和 benchmark 污染检查，然后按目标能力设计 mixture。

第三，设计训练配方。模型采用 decoder-only Transformer，目标是 causal LM loss。优化器用 AdamW，配合 warmup、cosine decay 或类似 schedule，设置 global batch size、gradient clipping、weight decay 和 mixed precision。

第四，设计分布式策略。根据模型大小和集群拓扑选择 data parallel、tensor parallel、pipeline parallel、ZeRO/FSDP 或组合策略，同时关注显存、通信和 pipeline bubble。

第五，训练监控。持续看 train loss、validation loss、LR、gradient norm、throughput、MFU、GPU utilization、NaN/Inf、checkpoint time 和分领域 eval。

第六，容错和 checkpoint。定期保存完整 checkpoint，包括模型、optimizer、scheduler、RNG、sampler 和 step 信息；resume 后检查 loss 连续性和状态一致性。

第七，评估和迭代。用通用 benchmark、领域 benchmark、人工样例、安全评估和污染检查判断模型是否变好。如果某个能力不足，再回到数据 mixture、训练 token、模型规模或后训练阶段迭代。
```

这个模板适合大多数 training 开放题。

## 6.20 准备清单

准备 Training 面试时，至少要能回答下面的问题：

1. 预训练的 loss 是什么，为什么 next-token prediction 有效？
2. 数据清洗、去重、质量过滤分别解决什么问题？
3. 数据 mixture 如何影响模型能力？
4. SFT 和预训练有什么区别？
5. AdamW、learning rate、warmup、batch size、gradient clipping 分别起什么作用？
6. DP、TP、PP、ZeRO/FSDP 分别解决什么问题？
7. 混合精度为什么能加速，风险是什么？
8. loss spike 可能有哪些原因，怎么排查？
9. 训练吞吐低怎么定位瓶颈？
10. checkpoint 应该保存哪些状态？
11. resume 后如何确认训练正确？
12. 如何估算一个模型的训练 FLOPs、时长和成本？
13. validation loss 和 benchmark 不一致时怎么分析？
14. 如何设计继续预训练提升领域能力？
15. 如何防止训练过程中的 silent bug？

这一章的核心不是让你背完所有训练系统名词，而是让你建立一个工程闭环：目标决定数据，数据决定能力上限，训练配方决定稳定性，分布式系统决定效率，监控和 checkpoint 决定能否安全跑完，评估决定下一轮怎么迭代。
