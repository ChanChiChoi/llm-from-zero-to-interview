# Step 3.5 Flash：S3F1、MTP-3 与 11B 激活参数

读 Step 3.5 Flash 的“45 层”“11B 激活参数”和“MTP-3”时，容易把三种不同的账混在一起：45 层描述骨干的深度，11B 近似描述每个 token 激活的骨干参数规模，MTP-3 则表示额外的未来位置预测头。它们分别影响信息可见范围、每 token 的前馈工作量和解码时的候选生成，不能把其中任何一个数字直接换成显存占用或吞吐倍数。

本章假设读者已经认识因果注意力、next-token prediction 和稀疏 MoE。一般 MoE 路由与通信见[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)和[第二十二章](22-Dense与MoE训练服务trade-off.md)，滑动窗口 pattern 见[第十九章](19-Sliding-Window与稀疏注意力.md)；这里聚焦 Step 3.5 的公开配置、实验和推测解码边界。架构事实以 [StepFun 技术报告 v2](https://arxiv.org/html/2602.10604v2) 与固定 revision 的 [Hugging Face 配置](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/8847e9698c01fa9b33ed6dc5e0b0c24ad9883686/config.json) 为主，模型卡与 vLLM 配方用于核对发布和部署说明。

## 80.1 先读参数账本：backbone 与 MTP 的口径

参数账本回答的是模型里有多少权重、一次 token 前向大约用到多少参数；它不是显存账本。内存还取决于权重精度与分片、KV cache、通信缓冲、临时工作区和并发请求。

### 80.1.1 45 层 backbone、196B 总参数与 11B 激活参数

技术报告把 Step 3.5 Flash 的 backbone 记为 196B 总参数、每 token 激活 11B 参数；公开配置列出 45 层、4096 hidden size、128,896 词表，模型卡称 context window 为 256K。这里的“激活参数”是每 token 的模型计算口径，不意味着只需存下 11B 权重，也不说明一次长请求需要多少显存。报告特别注明激活参数统计排除了 embedding 和 output matrices。配置中的最大序列长度是 262,144；这个位置上限不能单独证明模型在所有任务上都能有效利用整个长度。

### 80.1.2 42 个 MoE 层如何选择 288 个 routed experts 中的 top-8

45 层中，最前面 3 层的 FFN 是 dense，后 42 层采用 MoE。每个 MoE 层有 288 个 routed experts 和 1 个 shared expert；每个 token 选中 8 个 routed experts，模型卡称 shared expert 始终激活。因而，288 描述可选专家的容量，top-8 描述稀疏路由的专家计算；二者都不能直接推出权重如何分片或 token dispatch 的时间。专家并行的 all-to-all、跨 rank 负载不平衡和同步等待仍会影响端到端延迟，细节回链第二十一至二十二章的一般 MoE 说明。[技术报告](https://arxiv.org/html/2602.10604v2) 与[模型卡](https://huggingface.co/stepfun-ai/Step-3.5-Flash)分别给出层与路由信息。

### 80.1.3 MTP 参数如何进入总量与激活量

公开资料存在统计范围差异，读数时要保留来源口径：

| 统计口径 | 总参数 | 每 token 激活参数 | 来源说明 |
| --- | ---: | ---: | --- |
| backbone | 196B | 11B | 技术报告的主干账本；激活数排除 embedding/output |
| backbone 加 MTP-3 | 约 198B | 约 13B | 技术报告附录表对三个 MTP 模块的合并统计，按 B 为单位取整 |
| 模型卡参数表 | 196.81B，即 196B + 0.81B Head | 约 11B | 模型卡列出单个 0.81B head 口径，没有与报告的 MTP-3 合计值完全对齐 |

报告称每个 MTP head 增加约 0.81B 参数；三个 head 的精确和不能简单从模型卡的单个“Head”条目中读出。报告附录与模型卡的数字可以同时保留，但公开资料没有把所有统计范围逐项对账，因此不应擅自把模型卡数字改写成三头合计，也不应把 backbone 的 11B 当成运行全部 MTP 路径后的完整计算账。[报告附录表 6](https://arxiv.org/html/2602.10604v2) 与[模型卡参数表](https://huggingface.co/stepfun-ai/Step-3.5-Flash)支持这些分别标注的口径。

## 80.2 S3F1 排布怎样组织 45 层

S3F1 表示三个滑动窗口注意力层后接一个 full attention 层。单说“每四层一个 full”会漏掉序列的起始层；需要结合固定配置从第一层开始数。

### 80.2.1 首个 Full 层之后的 11 组 S3F1

报告和配置给出的顺序是：第 0 层先做 Full attention，此后重复 11 组“3 个 SWA + 1 个 Full”。所以 45 层可数为：

| 层编号 | 注意力类型 | 这一段如何组成 |
| --- | --- | --- |
| 0 | Full | 起始全注意力层 |
| 1–3 | SWA | 第 1 组的三个局部层 |
| 4 | Full | 第 1 组末尾 |
| 5–7 | SWA | 第 2 组的三个局部层 |
| 8 | Full | 第 2 组末尾 |
| … | … | 后续按同一四层组继续 |
| 41–43 | SWA | 第 11 组的三个局部层 |
| 44 | Full | 第 11 组末尾 |

合计为 33 个 SWA 层和 12 个 Full 层；FFN 类型与注意力类型是两条不同的层属性，前三个 dense FFN 层不等于前三个注意力层都属于同一种模式。配置的 layer_types 数组可直接核验实际起始位置和重复次序。[报告的架构图说明](https://arxiv.org/html/2602.10604v2)与[固定配置](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/8847e9698c01fa9b33ed6dc5e0b0c24ad9883686/config.json)一致。

### 80.2.2 512-token 窗口、GQA-8 与两类 query-head 数

SWA 的窗口大小为 $W=512$；Full attention 使用 8 个 KV heads，也就是 GQA-8 的 KV 头数。模型配置中 Full 层有 64 个 query heads，SWA 层增加到 96 个。query heads 决定有多少组查询表示，KV heads 决定 key/value 的分组与共享方式；GQA-8 不能被理解成“每层只有 8 个 query heads”。模型卡的 256K context 与 $W=512$ 的局部窗口也不是同一概念：前者是公开上下文上限，后者是某类层一次注意力运算直接访问的局部范围。

### 80.2.3 周期 Full 层怎样补充跨窗口信息路径

在因果掩码下，Full 层的当前位置可以直接读取此前所有位置；SWA 层只能直接读取窗口内的位置。但进入后续层的每个 hidden state 已经过前层计算，因此某个远处 token 的信息可能先在 Full 层进入一个位置的表示，再随这些表示经过后续局部层传递。这个跨深度路径让混合模型保留长距离信息通路，却不等于每一层都对完整前缀做一次注意力，也不保证精确检索质量。要判断实际能力，仍要看相应长度、任务和评测协议下的结果。

## 80.3 注意力计算账与混合结构的质量边界

SWA 减少局部层的可见 token 对数，周期 Full 层仍要处理完整前缀。先算可见 pair 数，能看出结构如何改变注意力交互规模；它还没有计入 head 数、投影、显存读写和 kernel 实现。

### 80.3.1 因果窗口的可见 pair 数怎样随长度增长

以下采用一个清楚的计数约定：每个 query 可以看见自己；Full 层看到从序列开头到当前位置，SWA 层最多看到包含当前位置在内的最近 $W$ 个 token。序列长度为 $n$ 时：

```math
\begin{aligned}
P_F(n) &= \sum_{i=1}^{n} i = \frac{n(n+1)}{2},\\
P_S(n,W) &= \sum_{i=1}^{n}\min(i,W),\\
P_S(n,W) &= \frac{W(W+1)}{2}+(n-W)W,\qquad n\ge W,\\
P_{\mathrm{45}}(n,W) &= 33P_S(n,W)+12P_F(n).
\end{aligned}
```

例如 $n=4096$、$W=512$ 时，每个 SWA 层有 $1{,}966{,}336$ 个可见 pair，每个 Full 层有 $8{,}390{,}656$ 个；45 层合计 $165{,}576{,}960$ 个。若 45 层全是 Full attention，则同一口径下为 $377{,}579{,}520$ 个，混合结构约为全 Full pair 数的 $43.9\%$。

这只是未加权的可见 pair 账本。它没有乘上每类层的 query-head 数，没有计算 Q/K/V 和输出投影、MLP、权重访存、KV 传输或 kernel 利用率；因此 $43.9\%$ 不是 FLOPs 比例，更不是“墙钟时间减少 $56.1\%$”的预测。模型的 SWA 与 Full query-head 数本身也不同。

### 80.3.2 选择 SWA 而非线性注意力的作者动机

技术报告把 prefill 和 decode 的约束分开讨论：混合注意力降低长 prompt prefill 中的二次计算；对 decode，作者看重标准注意力的 KV mask 语义更适合并行验证 speculative candidates。报告选择 512 窗口 SWA，也说明当时缺少线性注意力在 agentic long-context 任务上优于 SWA 的稳健经验依据。这些是作者为该模型给出的设计动机和证据判断，不足以证明线性注意力普遍不适合长上下文，或 SWA 对所有模型、任务和硬件都更快。[技术报告第 2.1 节](https://arxiv.org/html/2602.10604v2)描述了这一选择范围。

### 80.3.3 30B-A3B 注意力消融显示的成本—质量取舍

报告用 30B-A3B proxy 做布局消融：窗口固定为 $W=512$，训练总量约 1.4T tokens 后进行 SFT，且关闭 MTP，以隔离注意力结构的影响。其 attention-side FLOPs 分别对 prefill 和 decode 归一化到 S3F1 的 1.00；主要结果如下：

| 布局 | SWA query heads | 相对 attention-side FLOPs（decode / prefill） | LongCtx |
| --- | ---: | ---: | ---: |
| FFFF | 32 | 约 2.68 / 2.90 | 28.8 |
| S1F1 | 32 | 约 1.58 / 1.65 | 29.6 |
| S3F1 | 32 | 1.00 / 1.00 | 27.5 |
| S3F1 + Head | 48 | 约 1.01 / 1.02 | 28.2 |

这组实验显示，S3F1 的 attention-side FLOPs 最低，但该 proxy 的 LongCtx 指标也低于 FFFF；把 proxy 的 SWA query heads 从 32 增至 48 后，在少量额外 attention-side 计算下回升到 28.2。报告还指出 S1F1 的 LongCtx 最高，但成本明显更高。它支持的是特定训练与评测协议下的成本—质量取舍；它不是 196B 发布模型的端到端延迟对照，也不能证明“加头一定恢复质量”。[报告表 1 与附录 A.4](https://arxiv.org/html/2602.10604v2)列出实验设置、相对计算量和指标。

### 80.3.4 Prefill、decode、理论 FLOPs 与 wall-clock 延迟

Prefill 同时处理一段输入序列中的多个 query；Full attention 的可见 pair 会随输入长度平方增长，SWA 把每个 query 的直接窗口限制在 $W$ 个位置。Decode 则逐步处理新位置，但 Full 层仍要访问增长中的历史上下文。于是，prefill 的 pair 计数、decode 的 attention-side FLOPs、权重和 KV 的内存流量，以及端到端 wall-clock 延迟，是四种不同观测。

报告附录还给出理想 workload 下的相对 latency 模拟；模拟说明扩展 SWA heads 与 gate 在作者设定中的估算开销，不等于某个 runtime、硬件和 batch 上实测的服务收益。要把架构选择转成工程结论，必须固定模型 revision、上下文长度、batch/并发、精度、后端与 kernel，再分别测 TTFT、decode token latency 和吞吐。

## 80.4 两类注意力补偿：扩展 SWA query heads 与 head-wise gate

S3F1 减少了不同 token 之间直接连接的密度，报告因此增加局部 query heads，并引入逐头门控。这两项调整作用对象不同：扩头增加局部查询表示；门控按当前位置的信息动态调节一个头输出的强度。

### 80.4.1 从 64 到 96 个 SWA query heads

发布模型的 Full 层使用 64 个 query heads，SWA 层使用 96 个，而 KV heads 都是 8。更多 query heads 让局部注意力有更多查询通道，但它仍受 $W=512$ 限制。报告把扩头用于缓解 S3F1 相对全注意力基线的质量退化；“窗口固定”表示每个 query 的直接访问范围不随总序列长度增加，不表示增加的查询计算为零。报告附录在理想 workload 模拟中给出相对计算量和 latency 变化，不能替代特定部署的实测。

### 80.4.2 Head-wise gate 怎样抑制无用注意力输出

一个 attention head 先按注意力权重把可见位置的 value 向量合成为输出 $\boldsymbol{y}_i$。Step 3.5 再根据当前位置的输入表示 $\boldsymbol{x}_i$ 计算一个标量门控，并缩放该 head 的输出：

```math
g_i=\sigma(\boldsymbol{w}_{\mathrm{gate}}^{\top}\boldsymbol{x}_i),
\qquad
\boldsymbol{o}^{\mathrm{gate}}_i=g_i\boldsymbol{y}_i.
```

这里 $g_i$ 是当前位置、当前位置对应 head 的输入相关权重；当它较小时，整组注意力聚合结果对后续残差流的贡献也随之减小。它不是再次计算一遍 token 间注意力，也不是把窗口中的某个真实 token 删除。报告把这种输入相关的缩放解释为一种动态 sink mass：当窗口内注意力内容不值得充分吸收时，门控可以压低该头输出。相关等价解释来自作者的门控推导，正文只用它说明机制，不把它扩展成通用 gated-attention 定理。[技术报告附录 A.1](https://arxiv.org/html/2602.10604v2)给出了逐头公式和 sink 解释。

### 80.4.3 扩头、门控和 ideal-workload 模拟的证据边界

三类证据不要混成一个“结构提升”结论：

- 30B-A3B 的布局消融对照 Full、S1F1、S3F1 与扩头 S3F1，关闭 MTP；它回答布局与扩头在该 proxy 协议下的成本和质量变化。
- 100B-A10B 的 pre-training-only 对照在 S3F1 下比较 head-wise gate 与固定 sink token；报告所列平均指标为 64.4 与 62.5。这一结果针对该规模、数据和评测设置，不是最终 196B 模型的 gate 单因素评估。
- 附录的 64K/256K 延迟是理想 workload 模拟值。它描述作者模型下的相对开销，不说明任意框架都能达到相同比例。

跨表结果可以帮助理解设计动机，不能把不同模型规模、训练量和指标合并成单一因果估计。[报告第 2.3 节、表 2 与附录 A.2/A.4](https://arxiv.org/html/2602.10604v2)分别说明这些证据。

## 80.5 MTP-3 训练目标与三个未来偏移

MTP 有两个容易混淆的阶段：训练时，额外头让模型从同一序列位置学习更远的目标；推理时，只有 runtime 把这些头接入草稿—验证流程，它们才会减少目标模型的逐 token 调用。把多个位置作为训练目标，不自动意味着解码时会一次提交同样数量的 token。关于 MTP 作为辅助目标与推测解码草稿器的差别，也可参照 [Raschka 的 MTP 教学图解](https://sebastianraschka.com/llm-architecture-gallery/mtp/)；Step 3.5 的具体结构以其技术报告为准。

### 80.5.1 标准下一 token 与 MTP-1/2/3 的目标位置

在位置 $t$，普通语言模型头预测下一个 token $x_{t+1}$。报告把 MTP head 的编号定义为“超过标准 LM head 的额外偏移”：MTP-$h$ 预测 $x_{t+1+h}$。所以对应关系是：

| 预测头 | 从位置 $t$ 预测的目标 |
| --- | --- |
| 标准 LM head | $x_{t+1}$ |
| MTP-1 | $x_{t+2}$ |
| MTP-2 | $x_{t+3}$ |
| MTP-3 | $x_{t+4}$ |

三个 MTP 头代表三个额外的未来偏移；若把主 LM 头也计入训练目标，涉及四个预测位置。这个位置计数本身不说明任何一个候选在服务端被接受，也不规定每轮能提交几个 token。[技术报告第 2.2 节](https://arxiv.org/html/2602.10604v2)给出 MTP 的偏移定义。

### 80.5.2 SWA 与 dense FFN 构成的轻量 MTP 模块

技术报告说明，每个 MTP 模块由一个 SWA 和一个 dense FFN 组成，并以 backbone 在位置 $t$ 的 hidden state 为条件预测更远目标。相较于 backbone 的 42 个 MoE FFN 层，这条预测支路使用 dense FFN；它因此是独立于主干专家路由的计算路径，但仍要占用权重、执行注意力和前馈运算。轻量是相对主干规模的描述，不意味着零延迟或免费显存。

### 80.5.3 先训练 MTP-1，再初始化并联合微调 MTP-2/3

报告称主训练大部分阶段只训练 MTP-1；backbone 训练到后期后，再由 MTP-1 初始化 MTP-2 和 MTP-3，并在较轻量的末段共同微调。不同未来偏移的目标难度不同，报告因此按预测位置重加权 loss，以免优化过度偏向较远目标。公开报告说明了训练次序和原则，没有给出足以在此完整复现的逐阶段权重表；不应把未披露细节填成确定的训练配方。

## 80.6 从 MTP 预测到推测解码：草稿仍须经过目标验证

MTP 头可以提出未来 token，但验证器决定哪些候选进入正式输出。标准推测解码一次让草稿器生成短续写，再由目标模型并行打分，按从左到右的顺序接受候选；遇到首个拒绝时，就丢弃它之后依赖被拒 token 的剩余草稿。[Chen 等人的 speculative sampling 论文](https://arxiv.org/abs/2302.01318)说明了并行评分和修正采样；它支持通用机制，不是 Step 3.5 专属的运行结果。

### 80.6.1 Draft—verify—accept 的状态关系

先看一个贪心验证的简化轨迹。草稿器提出 A、B、C；目标模型在相同前缀下也选择 A，因此保留 A。加入 A 后，目标模型下一步选择 X，但草稿器此时给出 B，于是 B 被拒绝，系统输出目标选择的 X；C 是在错误候选 B 后面生成的，不能继续沿用。一次循环最后提交的是 A、X，随后以更新后的真实前缀重新起草。

随机采样模式下，需要修正拒绝规则才能保持输出分布与目标模型一致。令 $p$ 是目标分布、$q$ 是实际产生候选的草稿分布；候选 $y$ 的接受概率和拒绝后的替代分布为：

```math
\alpha(y)=\min\left(1,\frac{p(y\mid s)}{q(y\mid s)}\right),
\qquad
r(x\mid s)\propto\max\left(0,p(x\mid s)-q(x\mid s)\right).
```

这里 $s$ 是当前已确认前缀，$r$ 在词表上归一化后采样。每个候选只有在更早候选均已接受时才有资格验证；目标分布保持性属于满足相应采样与数值条件的标准算法，不是“任意比较两个 logits 都等价”。实际 MTP runtime 的拒绝、停止和缓存恢复细节仍应按具体实现核对。

### 80.6.2 接受长度影响每次目标调用的有效输出量

设一轮草稿长度为 $K$，目标模型接受的连续草稿前缀长度为 $A$。标准单轮算法在没有 EOS 或最大长度等提前终止时，拒绝位置会用一个校正 token 填上；如果全部草稿都被接受，则目标模型可以再给出一个 bonus token。因此通常可写为：

```math
N_{\mathrm{commit}}=A+1,\qquad 0\le A\le K.
```

这表示一次目标验证所提交的数量，不是 MTP 头数，也不保证达到上界。吞吐要同时看 draft 生成时间、目标并行验证时间、接受长度分布、同步与调度、并发和内存带宽。即使平均接受长度高，如果草稿成本或验证尾延迟太大，最终每秒输出 token 仍可能没有改善。

### 80.6.3 checkpoint 中三层 MTP 与 runtime 可用路径

固定模型配置把预测层数设为 3，代表 checkpoint 中有三层 next-token prediction modules。部署配置则是另一件事：截至 2026-11-25，StepFun 模型卡写明 vLLM 尚未提供完整 MTP-3 支持，并给出把 **num_nextn_predict_layers** 改为 1、草稿 token 数设为 1 的示例；vLLM 官方 recipe 同样配置一层、一个 speculative token。于是，“checkpoint 含三层”与“该 serving recipe 一次启用三层并验证多个 token”不能画等号。这里核验的是公开配置文件，没有在本地加载模型或测试 runtime，不据此宣称现有部署的全部支持状态。[模型卡部署说明](https://huggingface.co/stepfun-ai/Step-3.5-Flash)与[vLLM StepFun recipe](https://github.com/vllm-project/recipes/blob/main/StepFun/Step-3.5-Flash.md)给出了可复核的配置字段。

### 80.6.4 发布方 tokens/s 与框架示例 benchmark 的归因

模型卡报告常见使用约 $100$–$300\ \mathrm{tok/s}$，并称单流编码任务峰值约 $350\ \mathrm{tok/s}$。这是发布方性能陈述；该处没有充分给出可直接复算和横向对齐的硬件、批大小、量化、后端及任务细节，因此不能当作通用吞吐基线。vLLM recipe 展示的是随机输入/输出长度的 serving benchmark 示例，注明并发、TP 和示例硬件；它没有给出同一模型启用与关闭 MTP 的配对结果，所以无法单独测出 MTP 带来的增益。[模型卡](https://huggingface.co/stepfun-ai/Step-3.5-Flash)与[vLLM recipe benchmark](https://github.com/vllm-project/recipes/blob/main/StepFun/Step-3.5-Flash.md)分别支持上述发布主张和示例范围。

## 80.7 Agent benchmark 与 Context Manager 的归属

架构说明与 Agent benchmark 不是同一种证据。模型卡把 Context Manager 单独写进 BrowseComp 协议；理解这一点，才能区分模型内部结构与外部轨迹管理策略。

### 80.7.1 Context Manager 是超过阈值后重启 agent loop 的 harness 策略

模型卡对 Context Manager 的说明是：有效上下文超过预设阈值后，agent 重置上下文并重新开始 loop。这是 harness 的状态管理方式，不是模型参数、注意力机制或永久记忆。它会改变任务轨迹中哪些信息继续可见，也可能影响工具调用轮数和任务完成机会；模型卡只披露了此处的重启策略，不支持推断其具体摘要格式或内部保存了什么状态。

### 80.7.2 benchmark 结果不能替代架构因果证据

模型卡把 BrowseComp 的普通结果与启用 Context Manager 的结果分开列出（分别为 $51.6$ 与 $69.0$）。这直接提醒读者，带 harness 的分数会随轨迹管理协议而变；但这两项仍是发布方评测结果，不能说明分差由 SWA、MTP 或某一层结构单独造成。SWE-bench Verified、Terminal-Bench 2.0 等结果也要绑定任务集版本、工具权限、预算、超时、后端和复现方式。架构消融回答部件变化带来的差异，运行时 A/B 回答实现收益，带 harness 的 benchmark 回答系统在某协议下的结果，三者应分开报告。[模型卡评测表及注释](https://huggingface.co/stepfun-ai/Step-3.5-Flash)说明 Context Manager 的分数标签和重启规则。

## 80.8 面试追问

1. **45 层怎样数出 33 个 SWA 和 12 个 Full？** 从第 0 层的 leading Full 开始，再数 11 组“3 个 SWA + 1 个 Full”；说明注意力类型与前三层 dense FFN 是不同属性。
2. **为什么混合结构不是整体 $O(n)$？** 固定窗口层的 pair 数随 $nW$ 增长，但 12 个周期 Full 层仍各自处理完整因果前缀；理论 pair 比例也不能直接换算成端到端延迟。
3. **MTP-3 是不是每轮提交三个 token？** 说明三个额外目标偏移对应 $x_{t+2}$、$x_{t+3}$、$x_{t+4}$；目标验证、接受前缀和拒绝修正才决定实际输出数量。
4. **11B 激活参数为何不能换算成模型显存？** 分开解释完整权重、每 token 激活计算、KV cache、专家分片通信与运行时 workspace，并指出报告对 MTP-3 的参数统计和模型卡数字存在口径差异。
5. **什么证据能说明 S3F1 或 MTP 提升了速度？** 对齐模型、窗口、MTP 开关、上下文长度、任务、精度、硬件、后端和负载，分别测质量、接受长度、TTFT、decode latency 与吞吐；30B proxy 消融和无 MTP A/B 的 serving 示例不能合并成最终模型的实测加速结论。

## 80.9 小练习

1. 根据固定 revision 的 layer_types 还原 0–44 层的注意力类型，并核对 leading Full、11 组 S3F1、SWA/Full 总数和前三个 dense FFN 的位置。
2. 对 $n=4096$、$W=512$ 手算每层的 causal pair 数和 45 层总数；再改变 $n$ 和 $W$，解释哪些项线性增长、哪些项仍是二次增长。注明 pair 数遗漏了哪些实际计算与内存因素。
3. 画出普通 LM head 与 MTP-1/2/3 的目标位置；用报告和模型卡的原始参数口径重建参数表，列出仍无法从公开资料对账的部分。
4. 给一组草稿与目标分布，逐候选计算接受概率；在首个拒绝处计算残差分布，并说明为何后续候选必须丢弃。随后比较目标验证调用次数与已提交 token 数。
5. 设计固定 checkpoint、任务、量化、硬件、后端和并发的 MTP 开关 A/B；同时报告质量、接受前缀长度、draft/verify 延迟、TTFT、decode latency 和吞吐。另写出 BrowseComp 有无 Context Manager 的协议差异，避免把 harness 分差归因给注意力结构。
