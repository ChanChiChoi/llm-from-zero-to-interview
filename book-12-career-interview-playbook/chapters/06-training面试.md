# 第六章：大模型训练：从数据到可恢复系统

训练一个大模型，不是把数据交给一个循环，然后等待 loss 变小。它是一条由数据、目标、模型、优化器、分布式系统、数值表示、评估和恢复机制共同组成的生产链。任何一环的语义不清，都会让后面的数字失去解释力：数据重复会让有效 token 数被高估，padding 会让吞吐和 loss 被污染，通信等待会把理论算力变成低利用率，checkpoint 缺少 sampler 状态则可能让恢复后的训练悄悄改变数据顺序。

本章围绕一条可追溯的训练链展开：

~~~text
训练目标
    -> 数据契约与采样分布
    -> tokenizer、长度与有效 token
    -> 模型配置与计算预算
    -> 优化器、batch 和学习率
    -> 并行、显存与通信
    -> 混合精度与数值稳定
    -> 监控、评估与故障诊断
    -> checkpoint、恢复与成本
    -> 下一轮数据和配方迭代
~~~

初学者可以把每个环节理解成一个具体问题：模型究竟在学什么，哪些 token 真正参与了学习，一张 GPU 放不下时怎样切分，loss 异常时先查哪里，训练中断后如何证明恢复是正确的。有经验的读者还要继续追问：指标的分母是什么，变化是否能归因于某个改动，通信和 I/O 是否改变了有效预算，实验结果能否推广到目标分布，以及一个更好的 checkpoint 是否真的降低了单位成功能力成本。

第 5 章解释了概率、损失、Transformer 和 scaling 的基础；本章把这些概念放入一个可运行、可观测、可恢复的训练系统。第 7 章将继续讨论推理部署，因此本章只在需要时说明训练结果如何影响推理，不把部署实现提前压缩进来。

## 6.1 先定义训练对象，而不是先选择框架

### 6.1.1 训练系统到底在交付什么

训练的直接产物是 checkpoint，但 checkpoint 不是最终交付物。一个可用的训练项目至少要同时交付四类东西：

1. 模型状态：参数、词表、位置编码配置和结构配置能够彼此匹配。
2. 训练证据：数据版本、有效 token、loss、评估切片和实验配置可以追溯。
3. 系统证据：吞吐、显存、通信、故障和恢复行为有可复核记录。
4. 能力证据：目标任务有所改善，同时没有不可接受的通用能力、安全或事实性退化。

只保存一个 `model.safetensors` 文件，不能证明训练过程可复现，也不能证明模型获得了目标能力。相反，一个训练步数较少但数据、配置、评估和恢复证据完整的实验，往往比一个只给最终分数的长跑更容易继续迭代。

### 6.1.2 训练阶段改变的对象不同

预训练通常用大规模序列优化语言模型目标，使模型学习词法、结构、事实关联和跨文档模式。继续预训练仍然使用相近的语言模型目标，但把采样分布移动到新的领域、语言或时间范围。SFT 使用示范回答改变任务遵循、格式和交互行为。偏好学习进一步改变多个候选回答之间的选择倾向。

这些阶段可以使用相同的 Transformer，但它们优化的对象不同：

| 阶段 | 主要输入 | 直接优化对象 | 更可能改变什么 | 主要风险 |
| --- | --- | --- | --- | --- |
| 预训练 | 网页、书籍、代码、论文等序列 | 真实 token 的条件概率 | 基础语言和知识分布 | 数据质量、重复、污染、成本 |
| 继续预训练 | 领域或新时间段序列 | 新分布上的 token 概率 | 领域术语和分布适配 | 遗忘、领域偏置、过拟合 |
| SFT | 指令、回答、工具轨迹 | 示范目标 token 概率 | 格式、任务遵循、风格 | 模板过拟合、幻觉、误拒 |
| 偏好学习 | chosen/rejected 或比较信号 | 相对偏好目标 | 选择倾向、帮助性和安全行为 | 长度偏置、迎合、reward hacking |

“模型学到了新知识”必须结合阶段和证据来解释。SFT 当然可能让模型记住示范中的事实，但少量示范不等同于系统地覆盖一个领域；继续预训练可能提高领域验证 loss，却不自动带来正确的工具使用或安全边界。

### 6.1.3 训练契约的最小字段

在启动训练前，应把下面的字段写成版本化配置，而不是依赖口头约定：

- 目标阶段和目标能力；
- tokenizer、词表和 special token 版本；
- 最大序列长度、padding、packing 和 label mask 规则；
- 数据来源、过滤版本、去重范围和采样权重；
- 模型结构、参数精度、初始化和位置编码配置；
- global batch 的定义，是按样本还是按有效 token；
- optimizer、学习率、warmup、decay、梯度裁剪和 weight decay；
- 并行维度、设备拓扑和通信实现；
- checkpoint 内容、保存间隔、保留策略和恢复验证；
- validation、能力评估、安全评估和污染检查协议。

这些字段不是管理表格的装饰。比如只写“训练 100B tokens”还不够：需要说明是原始 token、过滤后 token、重复加权后的 token，还是实际参与 loss 的有效 token。只写“batch size 1024”也不够：还要知道它是每卡 micro-batch、全局样本数，还是每次更新聚合的 token 数。

## 6.2 数据不是文件集合，而是被采样的分布

### 6.2.1 从原始 artifact 到训练样本

一个网页、PDF、代码仓库或对话导出文件，不能直接等同于训练样本。可靠的数据管道通常要保留以下层次：

~~~text
原始 artifact
    -> 解析后的内容与结构
    -> 规范化文本和元数据
    -> 去重、质量、安全与隐私过滤
    -> tokenizer 结果和样本切分
    -> 混合采样清单
    -> 训练 shard 与可追溯索引
~~~

每层都应能回答三个问题：输入从哪里来，做了什么变换，输出如何回到原始来源。只保留最后的 token 二进制而删除来源和过滤记录，会让后续无法解释某类知识消失的原因，也无法在发现隐私或版权问题时精确定位删除范围。

### 6.2.2 数据质量至少有五个维度

“质量高”不是一个单一分数。对大模型训练，更有用的拆分是：

1. 信息质量：内容是否有事实、结构或可迁移模式，而不是大量导航、广告和乱码。
2. 语言质量：句法、编码、Unicode、标点和语言标签是否稳定。
3. 任务相关性：数据是否覆盖目标领域、格式、工具和推理分布。
4. 安全与隐私：是否存在个人信息、恶意代码、危险指令或不应进入训练的数据。
5. 评估独立性：样本或近邻是否与 validation、benchmark 和人工测试重叠。

过滤器只是测量工具，不是真值裁判。一个以困惑度为基础的小模型过滤器，可能删除口语、代码注释和低频语言；一个过强的重复过滤器，可能删除真实世界中本来就高频的 API 用法。过滤规则应按来源、语言和任务切片观察，并用消融实验确认它改变的是目标能力，而不只是数据规模。

### 6.2.3 精确去重、近似去重与污染检查

精确去重可以按规范化后的字符串、文件哈希或样本哈希删除完全相同的记录。近似去重需要把文本表示成 n-gram、MinHash、SimHash 或 embedding，再根据相似度聚类或过滤。两者解决的问题不同：前者成本低、边界清楚；后者能处理轻微改写，却更容易误删相关但有价值的样本。

评估污染是去重的另一面。训练集和测试集没有逐字相同，也可能存在高相似的网页副本、题目解析、代码镜像或模板。污染检查需要记录比较算法、阈值、时间范围和人工复核规则。一个“没有发现重叠”的结论，只能表示在该检查协议下没有发现，不等于数学意义上的完全独立。

### 6.2.4 混合比例决定模型看见什么

设数据源有 `K>0` 类，第 k 类的目标采样比例为 `w_k\geq0`，满足所有比例之和为 1。若总有效训练 token 为 `D>0`，则第 k 类期望被采样的 token 数近似为：

~~~math
D_k
\approx
w_kD,
\qquad
\sum_{k=1}^{K}w_k=1
~~~

实际数据集大小、重复采样、上采样和过滤损耗会使 `D_k` 与这个期望不同。更重要的是，能力不是数据比例的线性函数：增加代码数据可能改变代码能力、自然语言风格、token 分布和训练速度，不能只用“代码 token 占比提高了”解释结果。

网页提供覆盖面，书籍和长文提供段落结构，论文提供专业表达，代码提供严格的符号和执行反馈，数学数据提供形式化推导，对话和工具轨迹提供交互协议。它们不是互相替代的标签，而是不同的训练信号。数据混合应以目标能力和成本为约束，通过小规模对照而不是永久套用一个经验比例。

### 6.2.5 采样概率与数据量不是一回事

设第 k 类原始有效 token 数为 `S_k\geq0`，采样权重为 `a_k\geq0`，且至少有一个 `a_kS_k` 为正。如果按加权重采样，归一化后的概率为：

~~~math
p_k
=
\frac{a_kS_k}{\sum_{j=1}^{K}a_jS_j}
~~~

当某个小数据源被上采样时，它的 epoch 定义会变得模糊：同一条样本可能被重复看到多次。报告训练 token 时，应区分“抽取 token 数”和“去重后独立 token 数”；报告数据比例时，应说明是按原始数据、过滤后数据还是实际训练采样统计。

### 6.2.6 一个数据混合的数值例子

假设过滤后有网页 800B token、代码 150B token、数学 50B token。训练预算是 200B token，目标采样比例为 0.60、0.25、0.15，则期望抽取：

~~~math
D_{\mathrm{web}}=120\mathrm{B},
\qquad
D_{\mathrm{code}}=50\mathrm{B},
\qquad
D_{\mathrm{math}}=30\mathrm{B}
~~~

网页和代码可以主要来自一次遍历，数学则需要较高的重复或更积极的抽样。若数学样本实际只有 20B 且每条最多重复两次，那么“30B 数学 token”并不等于 30B 独立数学内容。这个区别会影响记忆风险、验证集独立性和对训练收益的解释。

## 6.3 Tokenizer、序列长度与 packing

### 6.3.1 模型真正处理的是有效 token

原始字符数、token 数、序列槽位数和参与 loss 的目标 token 数是四个不同统计量。一个 batch 可能有 1,000,000 个分配的槽位，但其中一部分是 padding，一部分是 prompt mask，真正贡献梯度的 token 可能只有 760,000。

令一个非空 batch 中第 i 条样本分配到的槽位数为 `c_i>0`，非 padding token 数为 `0\leq e_i\leq c_i`。padding 浪费率可以定义为：

~~~math
r_{\mathrm{pad}}
=
1-
\frac{\sum_{i=1}^{B}e_i}{\sum_{i=1}^{B}c_i}
~~~

这个定义把被 mask 的 prompt 是否算作有效 token 交给项目契约；这里的分子使用非 padding token，另行报告 loss 有效 token。分母必须说明，否则“利用率 80%”无法比较。

### 6.3.2 Packing 为什么能提高吞吐

若把多条短样本拼接到固定长度序列中，padding 可以减少，设备上的矩阵乘法更接近满载。但 packing 需要明确样本边界：不同文档之间是否允许 attention，EOS 是否加入，loss 是否跨边界计算，position id 是否重置。若把本不相邻的样本当作一个连续文档，模型可能学习到人为的跨样本依赖。

两种常见语义是：

- 独立 packing：每个样本有边界 mask，不能读取另一个样本的 token；
- 连续 stream packing：把文本视为同一训练流，只在文档边界加入分隔符。

它们的计算形状可以相同，训练语义却不同。实现时必须用小样本逐位置检查 attention mask 和 labels，而不能只比较 tokens/sec。

### 6.3.3 变长 batch 的 loss reduction

设一个非空 batch 中第 i 条序列有 `m_i\geq0` 个有效目标 token，每个 token 的损失为 `ell_{i,t}`。在 `\sum_i m_i>0` 时，按 token 平均的 loss 为：

~~~math
\mathcal{L}_{\mathrm{token}}
=
\frac{\sum_i\sum_{t=1}^{m_i}\ell_{i,t}}
{\sum_i m_i}
~~~

若每条序列都满足 `m_i>0`，按序列平均则为：

~~~math
\mathcal{L}_{\mathrm{seq}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(
\frac{1}{m_i}
\sum_{t=1}^{m_i}\ell_{i,t}
\right)
~~~

当 `m_i` 不相等时，两者并不相同；如果某条序列 `m_i=0`，它不能进入按序列平均的分母，应被显式跳过或作为无监督样本单独统计。前者让每个 token 贡献相近，后者让每条样本贡献相近。预训练通常关注 token-level 预算，指令数据有时更关心每条任务的均衡；选择哪一种不是实现细节，而是优化目标的一部分。

### 6.3.4 tokenizer 变化会改变训练预算

同样的字符文本使用不同 tokenizer，可能产生不同的 token 数、不同的序列边界和不同的监督粒度。token 数更多会增加训练计算、KV cache 读取和上下文截断概率；token 数更少则可能让单个 token 携带更复杂的语义，改变预测难度。

因此在多语言或代码训练中，不能只报告“每种语言采样了相同 token 数”。还应报告字符数或字节数、token 密度、样本长度、截断率和有效 loss token。否则某种语言可能因为 tokenizer 更昂贵而在相同 token 预算下获得更少的字符覆盖。

## 6.4 模型配置、显存与计算预算

### 6.4.1 参数量不是显存账本

训练显存至少由参数、梯度、优化器状态、激活和临时通信缓冲组成。设 `N>0`，参数和梯度每个元素分别使用正的 `b_w`、`b_g` 字节，优化器状态每个参数平均使用非负的 `b_o` 字节，激活显存为非负的 `M_act`，临时通信和 kernel 缓冲为非负的 `M_temp`，则粗略显存模型为：

~~~math
M_{\mathrm{train}}
\approx
N(b_w+b_g+b_o)
+M_{\mathrm{act}}
+M_{\mathrm{temp}}
~~~

这是容量估算，不是某个框架的精确分配结果。AdamW 的一阶和二阶矩通常使用较高精度保存，混合精度训练还可能保留 master weights；因此“模型用 BF16”并不表示训练状态全部只占 BF16 的一份。

### 6.4.2 激活显存和三个长度相关

激活显存主要受 batch、序列长度、hidden size、层数、是否保存反向所需中间结果和 attention 实现影响。对 dense Transformer，可把主要趋势写成：

~~~math
M_{\mathrm{act}}
\propto
B\times T\times L\times H
~~~

其中 `B` 是 micro-batch，`T` 是序列长度，`L` 是层数，`H` 是 hidden size。标准 attention 还会产生与 `T^2` 相关的 score 或概率中间量；memory-efficient attention 可以避免显式保存完整矩阵，但不能把算法的所有计算和带宽都变成线性。

激活重计算用额外计算换显存：前向只保存部分边界，反向时重新计算中间激活。它通常能让更大的 batch 或更长序列放入显存，但会降低单步速度。正确比较时要同时记录峰值显存、有效 tokens/sec 和总训练时间。

### 6.4.3 Dense 与 MoE 的参数账本不同

Dense 模型每个 token 都经过大部分参数。MoE 拥有较大的 total parameters，但路由器只把每个 token 发给部分 expert，因此每 token 的 active parameters 较少。训练和推理的计算、通信、负载均衡和 checkpoint 大小都需要分别记账。

如果只用 total parameters 估算 MoE 的每 token FLOPs，会高估计算；如果只用 active parameters 估算存储和优化器状态，会低估显存。模型规模、激活计算和部署成本必须放在不同列中。

### 6.4.4 训练 FLOPs 的数量级估算

设 dense decoder-only 模型参数量为 `N>0`，训练 token 数为 `D>0`，每个 token 的前向加反向计算系数为 `k>0`。一个教学级估算是：

~~~math
C_{\mathrm{train}}
\approx
kND
~~~

常见的 `k` 量级近似会取 6 左右，但它依赖前向/反向计数、词表投影、attention、重计算、序列长度和实现，不能当作硬件实测。通信、数据读取、评估、checkpoint 和故障重跑都不在这个公式里。

### 6.4.5 一个 7B 与 1T token 的计算例子

设 `N=7\times10^9`、`D=10^{12}`，先用 `k=6` 做数量级估算：

~~~math
C_{\mathrm{train}}
\approx
6\times7\times10^9\times10^{12}
=
4.2\times10^{22}\ \mathrm{FLOPs}
~~~

假设单卡理论峰值为 `300 TFLOPs`，实际 MFU 为 0.40，则每卡有效算力约为 `1.2\times10^{14} FLOPs/s`。使用 1,024 张卡时，有效集群算力约为 `1.2288\times10^{17} FLOPs/s`，理想计算时间为：

~~~math
t_{\mathrm{ideal}}
\approx
\frac{4.2\times10^{22}}
{1.2288\times10^{17}}
\approx
3.42\times10^5\ \mathrm{s}
\approx
95\ \mathrm{h}
~~~

真实排期还要乘上数据等待、通信、checkpoint、评估、故障和利用率波动的系数。这个例子最有价值的地方不是“95 小时”这个数字，而是把参数量、token 预算、有效算力和非计算开销放在同一个账本里。

## 6.5 优化配方：让每个 token 产生有意义的更新

### 6.5.1 Global batch 应按更新语义定义

设每个设备的 micro-batch 为正整数 `B_micro`，数据并行副本数为正整数 `N_dp`，梯度累积步数为正整数 `K_accum`。在等长、没有丢弃样本的简化场景中，global sample batch 为：

~~~math
B_{\mathrm{global}}
=
B_{\mathrm{micro}}N_{\mathrm{dp}}K_{\mathrm{accum}}
~~~

语言模型更稳妥的定义是 global effective tokens：

~~~math
T_{\mathrm{global}}
=
\sum_{\substack{\text{all devices}\\\text{all accumulation steps}}}
\text{有效目标 token 数}
~~~

在变长、packing 和 prompt mask 存在时，样本数相同不意味着梯度权重相同。若实现按每个 micro-batch 的平均 loss 再平均，可能把短 batch 和长 batch 等权，导致与全局 token 平均不一致。

### 6.5.2 AdamW、学习率和 token-based schedule

AdamW 使用梯度的一阶和二阶统计，再将 weight decay 作为独立参数衰减。训练配置中真正影响轨迹的，不只是初始学习率，还包括 warmup 长度、decay 总步数、beta、epsilon、weight decay、梯度裁剪和有效 batch。

对不同序列长度或不同 batch size 的实验，按 optimizer step 定义 warmup 可能造成不同的 token warmup 预算。更容易比较的方式是记录 `seen_tokens`，并用 token 数驱动 schedule：

~~~math
\eta(s)
=
\begin{cases}
\eta_{\max}\dfrac{s}{S_{\mathrm{warm}}}, & 0\le s<S_{\mathrm{warm}} \\
\eta_{\mathrm{decay}}(s), & s\ge S_{\mathrm{warm}}
\end{cases}
~~~

其中 `s` 是已经处理的有效 token 数，`S_warm` 是 warmup token 数。具体的 decay 可以是 cosine、linear 或其他调度，但必须记录它的定义和恢复状态。

### 6.5.3 梯度裁剪是保护措施，不是修复器

设聚合后的梯度向量为 `g`，其 L2 范数为 `||g||_2`，裁剪阈值为 `c\geq0`，并取 `\varepsilon>0`。全局范数裁剪可以写成：

~~~math
g'
=
g\cdot
\min\left(1,\frac{c}{\lVert g\rVert_2+\varepsilon}\right)
~~~

它可以限制异常 batch 对参数的单步影响，但会改变梯度方向的尺度，不能修复 label shift、错误 mask、NaN 传播或 checkpoint 不一致。记录裁剪前后的梯度范数和被裁剪比例，才能知道它是在偶尔保护训练，还是长期掩盖配方问题。

### 6.5.4 训练状态不等于参数状态

同一组参数配上不同的 AdamW 动量、学习率位置、数据顺序和随机数状态，下一步更新可能不同。因此训练实验必须把参数、optimizer、scheduler、RNG、sampler 和配置视为一个状态整体。只比较两个最终参数文件，无法判断它们是相同训练轨迹的不同结果，还是从不同状态开始的两个实验。

## 6.6 分布式训练：切分什么，通信什么

### 6.6.1 Data Parallel：复制模型，切分数据

Data Parallel 在每个副本保存一份模型，不同副本处理不同样本，再通过 collective communication 聚合梯度。它最直观，适合模型和训练状态能放入单个副本、主要希望扩大 batch 的场景。

它的代价是参数、梯度和 optimizer state 被重复保存；当模型变大时，显存先成为限制。梯度 all-reduce 还会占用网络带宽，扩展到更多卡后，单步时间不一定按卡数下降。

### 6.6.2 Tensor Parallel：切分层内矩阵

Tensor Parallel 把线性层或 attention 投影的矩阵按行、列或 head 维度切分，使单层参数和中间张量分布在多张卡上。它解决的是单层放不下和层内计算规模过大的问题。

层内切分意味着前向和反向频繁通信，通常要求同一节点或高速互联域。TP degree 增大后，计算量下降，但 collective 的延迟和同步频率上升；选择切分维度时还要保证 hidden size、head 数和矩阵维度能整除。

### 6.6.3 Pipeline Parallel：切分层与时间

Pipeline Parallel 把连续层放到不同 stage，让多个 micro-batch 在流水线中同时运行。它能容纳更深的模型，但存在 pipeline bubble：流水线填充和排空时，部分 stage 没有工作。

若 pipeline 有 `P` 个 stage，使用 `M` 个 micro-batch，理想化的 bubble 比例会随 `P/M` 增大而变差；具体调度还取决于 1F1B、交错 stage、负载不均衡和通信。增加 micro-batch 可以减少空泡，却会改变 activation、梯度累积和 latency。

### 6.6.4 ZeRO 与 FSDP：切分训练状态

ZeRO 和 FSDP 的核心思想是避免每个 data-parallel 副本都保存完整的参数、梯度和优化器状态。不同实现和 stage 的切分范围不同，常见对象包括：

1. optimizer state；
2. gradients；
3. parameters；
4. 参数在前向和反向期间临时 all-gather 的生命周期。

显存下降通常伴随通信增加。参数切分不是“免费显存”，实际性能取决于 gather/reduce-scatter 是否能和计算重叠、网络拓扑、bucket 大小和模块粒度。一个配置在小模型上看起来更省显存，放到大模型或不同拓扑上可能因为通信而变慢。

### 6.6.5 Sequence Parallel 与 Context Parallel

当序列很长时，单个样本沿序列维度产生大量 activation。Sequence Parallel 可以在 tensor-parallel 组内切分部分序列相关计算；Context Parallel 则更广泛地沿上下文维度分布 attention 或状态。它们要处理跨设备的 key/value、mask、position 和通信。

长上下文训练的困难不只是把 `T` 乘大：attention 可见性、位置外推、激活内存、通信体积和数据 packing 都会同时变化。只看模型配置中的 `max_position_embeddings`，不能证明训练系统和模型能力已经支持该长度。

### 6.6.6 并行度与设备拓扑

若 tensor、pipeline 和 data parallel 的并行度分别为正整数 `P_tp`、`P_pp`、`P_dp`，总设备数在简单笛卡尔组织下近似为：

~~~math
N_{\mathrm{device}}
=
P_{\mathrm{tp}}P_{\mathrm{pp}}P_{\mathrm{dp}}
~~~

这个等式只描述布局，不描述效率。TP 通常希望落在高速 NVLink 域，PP 可以跨节点但需要稳定的 stage 间通信，DP 的梯度同步可能跨更大网络。设备数量、rank 映射、NUMA、网络拓扑和存储路径必须作为一个系统设计来验证。

### 6.6.7 通信与计算重叠

训练 step 可以拆成计算、通信、数据和保存时间：

~~~math
t_{\mathrm{step}}
\approx
t_{\mathrm{compute}}
+t_{\mathrm{comm}}
+t_{\mathrm{input}}
+t_{\mathrm{save}}
~~~

如果通信能够完全与计算重叠，实际时间更接近各阶段关键路径的最大值；如果同步点过多或 bucket 太小，则近似相加。优化前先用 profiler 测出时间线，才能知道应该改变并行度、bucket、数据读取，还是 kernel。

## 6.7 混合精度与数值稳定性

### 6.7.1 FP32、FP16、BF16 与 FP8 的取舍

不同浮点格式在符号位、指数范围和尾数精度上不同。FP32 范围和精度较宽但占用大；FP16 节省内存却更容易 overflow 或 underflow；BF16 的指数范围接近 FP32，通常更能承受大模型训练中的动态范围，但尾数更短；FP8 进一步降低存储和带宽，却需要更细致的缩放、校准和硬件支持。

“使用 BF16”通常只描述部分 forward/backward 张量，不代表 optimizer state、master weights、归约缓冲和 checkpoint 都使用同样精度。训练报告应分别写清参数、梯度、激活、optimizer state 和通信缓冲的 dtype。

### 6.7.2 Loss scaling 为什么存在

FP16 的小数范围有限，梯度很小时可能下溢为 0。loss scaling 先把 loss 乘以尺度 `s`，反向得到缩放梯度，再在更新前除以 `s`。如果检测到 Inf/NaN，则跳过该步并降低尺度；若长期稳定，可以逐步提高尺度。

缩放只能处理表示范围问题，不能修复错误的梯度。BF16 通常不需要与 FP16 相同的动态 loss scaling，但仍可能出现 overflow、异常激活和归约误差。

### 6.7.3 稳定 softmax 与归约

对 logits 向量 `z`，直接计算 `exp(z_i)` 可能溢出。稳定 softmax 先减去最大值 `m=max_i z_i`：

~~~math
\operatorname{softmax}(z_i)
=
\frac{\exp(z_i-m)}
{\sum_j\exp(z_j-m)}
~~~

这不会改变数学结果，却能显著改善数值范围。类似地，loss、梯度范数和 all-reduce 的累加精度都会影响大规模训练。一个 rank 上的非有限值如果没有及时同步，可能在数步之后才变成全局异常。

### 6.7.4 数值异常的来源

NaN 或 Inf 可能来自：

- 学习率、初始化或残差尺度过大；
- FP16 overflow、梯度下溢或 loss scaling 失配；
- attention score、softmax、归一化或除法中的非法值；
- 输入数据含有异常 Unicode、极端长度、非法 label 或空有效目标；
- mask 把所有 key 屏蔽后继续做不安全的 softmax；
- 某个 rank 的通信或 optimizer state 损坏；
- 恢复时 dtype、参数组或 scheduler 状态不匹配。

排查时要记录首次出现非有限值的位置，而不只记录最终 loss。可以逐层检查参数、激活、梯度和 optimizer state，并保存触发 batch 的原始样本和 mask。越接近第一次异常，证据越有价值。

## 6.8 一个训练 step 的语义

### 6.8.1 Causal LM 的标签错位

对输入 token 序列 `[x_0, x_1, ..., x_T]`，位置 `0` 的 logits 预测 `x_1`，位置 `T-1` 的 logits 预测 `x_T`。因此常见实现使用 `logits[:, :-1]` 对齐 `labels[:, 1:]`。如果错位，模型可能在训练中看到错误目标，loss 仍然能计算，却不再代表预期的 next-token 任务。

有效 label 的 mask 要与训练阶段一致。SFT 可能只对 assistant response 计算 loss，padding 通常使用 `ignore_index`，packing 则需要处理样本边界。一个“loss 正常下降”的曲线不能单独证明 shift 和 mask 正确。

### 6.8.2 一个可复核的最小 loss 实现

下面的实现刻意把 shift、ignore mask 和全忽略样本检查写出来。它适合作为训练框架中 fused loss 的参考测试，而不是用于替代高性能 kernel：

~~~python
import torch
import torch.nn.functional as F


def causal_lm_loss(logits, input_ids, ignore_index=-100):
    """Return the mean loss over valid next-token labels."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, time, vocab] and ids [batch, time]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share batch and time dimensions")
    if input_ids.shape[1] < 2:
        raise ValueError("at least two positions are required")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = shifted_labels.reshape(-1).ne(ignore_index)
    if not bool(valid.any()):
        raise ValueError("no valid target token")
    return per_token[valid].mean()
~~~

这个函数的检查对应三个契约：时间维度必须对齐，标签必须向左移动一个位置，平均值只能在有效目标上计算。生产实现还要测试分布式 loss 的全局 token 归约、混合精度和 fused kernel 与参考实现的数值误差。

### 6.8.3 梯度累积的正确归一化

设第 j 个 micro-batch 的有效 token 数为 `m_j`，该批次所有有效 token 的损失和为 `s_j`。如果希望整个 accumulation cycle 按 token 平均，最终 loss 应为：

~~~math
\mathcal{L}_{\mathrm{cycle}}
=
\frac{\sum_j s_j}
{\sum_j m_j}
~~~

简单计算每个 micro-batch 的平均 loss 再除以 micro-batch 数，等价于：

~~~math
\frac{1}{K}
\sum_{j=1}^{K}
\frac{s_j}{m_j}
~~~

这只有在各个 `m_j` 相等时才与 token 平均一致。对于变长数据，错误归一化会改变梯度方向，且会随着 padding 比例变化而变化。

### 6.8.4 训练状态机

一次可靠的训练更新可以抽象为：

~~~text
读取并验证 batch
    -> forward
    -> shift 与 mask
    -> loss reduction
    -> backward
    -> 梯度聚合与裁剪
    -> optimizer/scheduler 更新
    -> 记录指标
    -> 周期性评估与 checkpoint
~~~

每个箭头都可能改变状态。比如梯度裁剪发生在 all-reduce 前后，结果不同；scheduler 在 micro-step 还是 optimizer-step 更新，学习率轨迹不同；checkpoint 在参数更新前还是更新后保存，恢复时的 step 语义不同。训练框架应明确这些边界，并用小规模 reference test 固定行为。

## 6.9 监控：把运行状态变成证据

### 6.9.1 Loss 不是训练仪表盘

至少要同时观察以下指标：

| 指标 | 主要回答的问题 | 需要的上下文 |
| --- | --- | --- |
| train loss | 训练分布上的拟合是否变化 | tokenizer、mask、reduction |
| validation loss | 保留分布上的拟合是否变化 | 切片、污染、token 统计 |
| effective tokens/sec | 真正处理了多少目标 token | padding、packing、mask |
| step time | 一次更新耗时在哪里 | compute、comm、I/O、save |
| gradient norm | 更新信号是否异常 | clipping、dtype、rank |
| update/weight ratio | 参数改变幅度是否合理 | optimizer、参数组 |
| MFU | 理论 FLOPs 被利用了多少 | 模型 FLOPs 定义、硬件 |
| eval slices | 哪类能力变好或变差 | 样本量、协议、置信区间 |

单个指标很容易误导。例如 train loss 降低而 effective tokens/sec 降低，可能是过滤或 packing 改变；GPU utilization 高而 MFU 低，可能是 kernel 在做大量低效工作；validation loss 稳定而代码能力下降，可能是验证集不覆盖目标结构。

### 6.9.2 有效吞吐的定义

设一个 step 中实际经过设备的非 padding token 数为 `T_nonpad\geq0`，真正参与目标 loss 的 token 数为 `T_loss\geq0`，并且 `t_step>0`。可以同时报告：

~~~math
R_{\mathrm{compute}}
=
\frac{T_{\mathrm{nonpad}}}{t_{\mathrm{step}}},
\qquad
R_{\mathrm{loss}}
=
\frac{T_{\mathrm{loss}}}{t_{\mathrm{step}}}
~~~

前者更接近设备计算负载，后者更接近训练目标推进速度。若 prompt token 不计算 loss，二者可能明显不同。只报告 samples/sec 会掩盖变长 batch、padding 和模板 mask 的影响。

### 6.9.3 MFU 是模型定义下的指标

设每个 token 的理论训练 FLOPs 估算为 `f_model>0`，有效训练 token 吞吐为 `R_token\geq0`，设备峰值 FLOPs 为 `F_peak>0`，设备数为正整数 `N_device`，则教学化 MFU 为：

~~~math
\operatorname{MFU}
=
\frac{f_{\mathrm{model}}R_{\mathrm{token}}}
{N_{\mathrm{device}}F_{\mathrm{peak}}}
~~~

如果 `f_model` 用了不同的 attention、MoE 或重计算假设，两个团队的 MFU 不能直接比较。MFU 也不包括数据等待和保存时间，项目排期仍应使用端到端有效吞吐。

### 6.9.4 指标切片比一个总平均更重要

至少应按数据来源、语言、长度、任务类型、时间范围和安全类别切片。一个总体 validation loss 可能由占比很高的网页短文本主导，而目标业务的长文、代码或低资源语言正在退化。

切片不是越多越好。切片应提前注册，避免只挑选改善最大的分组；每个切片要记录样本数、有效 token 数、版本和不确定性。对于小切片，单次变化可能只是抽样波动，不能直接当作能力回归。

## 6.10 Loss spike：从症状回到根因

### 6.10.1 先描述 spike 的形状

“loss spike”至少有四种不同形状：

1. 单个 batch 的短暂尖峰，随后回到原趋势；
2. 某个阶段开始后持续升高；
3. 所有 rank 同时异常；
4. 只有一个或少数 rank 异常，随后污染全局。

形状决定排查顺序。单点尖峰优先看 batch、长度和 mask；在 warmup 或学习率边界发生的持续异常优先看 schedule；只有部分 rank 异常优先看分片、通信、硬件和数据随机性；恢复后突然改变则优先核对 checkpoint 状态。

### 6.10.2 数据原因

数据异常可能是坏编码、极端长序列、重复样本、错误标签、空有效目标、异常数字范围或未预期的控制 token。排查要保存触发 spike 的样本 ID、来源、长度、token 统计、mask 比例和前后若干 batch，而不是只保存一个 loss 数字。

若 spike 总是在某个来源或某类长度出现，可以把样本重新送入 CPU reference tokenizer 和 loss，检查 tokenizer 版本、截断、padding、position id 和 label shift。若 reference 正常而 fused 路径异常，问题更可能在 kernel、dtype 或分布式归约。

### 6.10.3 优化和数值原因

检查学习率、warmup token、optimizer step、gradient norm、clip fraction、参数 norm、update/weight ratio、NaN/Inf 和 loss scale。一个常见误区是把调小学习率作为唯一处理；如果根因是错误 mask 或损坏 optimizer state，调小学习率只会延迟下一次故障。

恢复时还要检查 scheduler 是否跳步、梯度累积计数是否重置、optimizer state 是否只加载了部分 shard，以及不同 rank 是否从同一训练版本恢复。

### 6.10.4 分布式和硬件原因

如果一个 rank 先出现非有限梯度，默认的 collective 可能把异常传播给全部 rank。需要记录每个 rank 的局部 loss、梯度范数、数据 shard、通信错误和设备健康状态。GPU 利用率突然归零、ECC/Xid 错误、网络重传或 NCCL timeout 可能与模型数学无关，却会表现为训练停顿或状态不一致。

### 6.10.5 一个完整的 spike 复盘

假设“云砺”训练任务在第 18,420 个 optimizer step 出现 loss 从 2.8 跳到 9.4，两个 step 后变成 NaN。初步日志显示 GPU 利用率正常，只有 rank 7 的梯度范数先超过阈值。

第一轮检查发现 rank 7 的 batch 中有一条长度等于上限的样本，padding mask 的 key 轴全部为屏蔽值，fused softmax 在该行产生了非有限结果。回退到最近 checkpoint 并删除这条样本后训练恢复，但这还不是完整修复，因为同类样本仍可能再次出现。

第二轮修复增加了全屏蔽行的显式处理、输入 batch 断言和触发样本保存；CPU reference 与 fused attention 对随机 mask 做了对照。回归训练后，正常样本的 loss 与原轨迹一致，异常样本被记录并进入数据清理队列。这个案例说明，loss spike 的处理包含缓解、定位、修复和回归四个不同动作，不能以“训练继续跑了”作为结束。

## 6.11 评估：训练 loss 之外的能力证据

### 6.11.1 Validation loss 测量什么

Validation loss 测量模型对一个保留数据分布的 token 预测能力。它适合观察训练是否继续拟合该分布，也适合比较配方在固定协议下的变化，但不直接测任务成功、工具安全或引用正确。

验证集应固定版本、避免训练污染，并按领域和长度切片。若训练数据持续变化，验证集也要检查是否被新数据近邻覆盖。一个经过反复调参的 validation set 不再是完全独立的最终证据，应保留更少被使用的 holdout。

### 6.11.2 能力评估需要协议契约

一个 benchmark 分数由模型、prompt、模板、解码、样本、评分器和版本共同决定。评估记录至少包括：

- 题目和答案版本；
- system/user 模板和 special token；
- temperature、top-p、最大输出长度和停止条件；
- 是否允许工具、检索或多次采样；
- 自动评分规则、人工评分规则和无效输出处理；
- 样本量、随机种子、失败样例和置信区间；
- 训练污染和近邻检查。

如果只保存一个平均分，后续无法判断分数变化来自模型能力、评估代码还是 prompt 变化。

### 6.11.3 二项任务的简单不确定性

对一个有 `n>0` 个独立评估样本、`0\leq k\leq n` 个正确样本的二项任务，准确率估计为：

~~~math
\hat p=\frac{k}{n}
~~~

当 `n` 不大或 `p` 接近 0/1 时，不能只用对称的正态近似。可以使用 Wilson 区间或精确二项区间报告不确定性。即使不写出区间公式，也要说明样本量和统计方法；一个 20 题任务从 70% 变成 75%，证据强度与 10,000 题任务完全不同。

### 6.11.4 评估矩阵而不是单一总分

一个训练改动至少要观察目标能力、通用能力、事实性、格式、拒答、安全和资源指标。比如增加数学数据可能提升数学 benchmark，却改变自然语言 validation loss 和输出风格；更强的 SFT 模板可能提升格式遵循，却增加正常问题的误拒。

评估矩阵的价值在于把收益和副作用同时显示出来。没有任何一个总分可以替代对高风险切片的人工复核和失败样例分析。

## 6.12 Checkpoint：把昂贵状态保存成可恢复证据

### 6.12.1 完整 checkpoint 应保存什么

可靠的训练 checkpoint 通常包含：

1. 模型参数和结构配置；
2. optimizer state，包括 AdamW 的一阶和二阶统计；
3. scheduler state、当前 optimizer step 和已处理 token；
4. gradient scaler 或混合精度状态；
5. Python、框架、CUDA 和各 rank 的 RNG state；
6. data sampler、epoch、shard 和样本游标；
7. tokenizer、数据版本、代码版本和训练配置；
8. 分布式切分、rank 映射和 shard manifest；
9. 最近的指标摘要、数据校验和完整性信息。

只保存参数适合发布推理模型，却通常不足以无缝恢复训练。缺少 sampler 状态可能导致数据重复或跳过；缺少 scheduler 状态会改变学习率；缺少 RNG 状态会改变 dropout 和数据增强轨迹。严格逐 bit 复现和语义等价恢复是两种不同目标，都应明确声明。

### 6.12.2 Checkpoint 频率的成本模型

设 `T_total>0`，checkpoint 间隔 `I>0`，单次保存耗时 `t_ckpt\geq0`，故障发生率 `lambda_fail\geq0`（单位时间内的期望故障次数）。在故障近似独立且 checkpoint 间隔固定时，期望故障次数约为 `lambda_fail T_total`，每次故障平均需要重算约 `I/2` 的训练时间。因此：

~~~math
T_{\mathrm{recompute}}
\approx
\lambda_{\mathrm{fail}}T_{\mathrm{total}}
\frac{I}{2}
~~~

保存本身还会带来近似的写入开销：

~~~math
T_{\mathrm{save}}
\approx
\frac{T_{\mathrm{total}}}{I}t_{\mathrm{ckpt}}
~~~

保存越频繁，`T_save` 占比越高；保存越稀疏，故障后的重算越多。实际最优点还受异步保存、故障相关性、存储带宽、checkpoint 大小和可接受恢复时间影响。频率不应只按“每 N 个 step”决定，也可以按有效 token、预计故障风险和里程碑评估决定。

### 6.12.3 原子写入与完整性

保存过程应避免让恢复程序看到半写入目录。常见做法是写入临时 shard，完成 fsync 或对象存储确认后生成 manifest，再原子更新一个指向新版本的指针。manifest 应包含每个 shard 的大小、哈希、参数范围和版本。

恢复前先验证所有 shard，而不是加载到一半才发现损坏。分布式保存还要确认 rank 数、切分方式和当前代码支持的 state dict 格式匹配。能读出参数并不等于参数被正确映射到了对应 rank。

### 6.12.4 Resume 后的连续性检查

恢复任务启动后，先在少量固定 batch 上比较：

- 参数和 optimizer state 的摘要；
- learning rate、seen tokens 和 step；
- loss、梯度范数和输出 logits；
- sampler 的下一个样本或 shard 位置；
- 不同 rank 的状态一致性。

如果追求严格复现，可以在相同 batch 上比较 logits 和梯度；如果只追求语义等价，则允许随机数和归约顺序带来的微小差异，但仍要设置数值误差范围和后续 loss/评估回归标准。

## 6.13 成本与容量：从 FLOPs 到单位能力

### 6.13.1 计算成本不等于账单成本

设租用单价为 `p_gpu`，GPU 数为 `G`，端到端训练小时数为 `h`，存储、网络、人工和失败重跑成本为 `C_other`，则粗略账单为：

~~~math
C_{\mathrm{bill}}
\approx
p_{\mathrm{gpu}}Gh+C_{\mathrm{other}}
~~~

如果只用理想 FLOPs 除以峰值算力，通常会低估时间；如果只看 GPU 小时，又可能忽略数据准备、评估、checkpoint 和多次实验。训练预算应同时记录预估、实际和偏差原因。

### 6.13.2 单位能力成本

假设某个训练候选在 `N_task>0` 个目标评估任务上的成功率为 `0\leq q\leq1`，总成本为 `C\geq0`。当 `q>0` 时，一个简单的单位成功成本是：

~~~math
C_{\mathrm{per\ success}}
=
\frac{C}{N_{\mathrm{task}}q}
~~~

其中 `N_task` 是评估任务数。如果 `q=0`，单位成功成本未定义，应报告“没有成功任务”，不能用零代替。这个指标不是产品收入，也不能替代安全和质量约束，但它提醒我们：一个更大的模型如果只带来很小的成功率增益，可能不值得增加训练和服务成本。

### 6.13.3 训练排期的分解

一个项目的日历时间至少包括：

1. 数据解析、过滤和 tokenizer 预处理；
2. 小规模配方和 kernel benchmark；
3. 主训练的有效计算时间；
4. 评估、人工分析和污染检查；
5. checkpoint 写入、故障恢复和重跑；
6. 发布前的模型转换和回归。

小规模 benchmark 的价值不只是预测 FLOPs，还能发现并行拓扑、packing、kernel、checkpoint 和数据读取的问题。用一段短跑测得的 end-to-end tokens/sec 乘到主训练预算，通常比直接套理论峰值更可信。

## 6.14 Scaling 与 compute-optimal：趋势不是处方

### 6.14.1 三个尺度要一起观察

模型参数量、训练 token 数和计算预算相互耦合。固定参数增加数据，可能改善欠训练；固定数据增加参数，可能提高容量但收益递减；固定算力时，参数和 token 的分配存在一个依赖模型、数据和目标的有效区域。

经验 scaling law 可以写成教学上的幂律近似：

~~~math
L(N,D,C)
\approx
L_\infty
+A N^{-\alpha}
+B D^{-\beta}
+E C^{-\gamma}
~~~

`N` 是参数量，`D` 是训练 token 数，`C` 是计算预算，`L_infty`、系数和指数由特定实验拟合。这个关系用于比较趋势和分配实验预算，不是所有任务、架构和后训练阶段都成立的物理定律。

### 6.14.2 Compute-optimal 的含义

Chinchilla 一类研究强调，在固定计算预算下，很多早期模型相对参数量使用了过少训练 token；增加高质量数据可能比继续增大参数更有效。这个结论依赖 dense 模型、数据质量、训练目标和评估指标，不能机械应用到多模态、MoE、长上下文或偏好学习。

当前项目应先写清“最优”针对什么：预训练 validation loss、代码能力、领域事实性、单位成功任务成本，还是上线收入。不同目标可能需要不同的参数、数据和后训练分配。

### 6.14.3 数据质量会改变 scaling 曲线

如果新增 token 大量重复、错误或与目标分布无关，名义上的 `D` 增加不等于有效数据增加。可以把有效数据质量用一个项目内的相对权重表示，但不能假设存在跨语料通用的固定质量系数。更可靠的做法是固定模型和训练预算，做数据来源、重复率、过滤强度和混合比例的对照。

## 6.15 继续预训练与 SFT 的训练设计

### 6.15.1 继续预训练适合改变领域分布

已有 base model 需要适应医学、法律、金融、代码库或企业术语时，继续预训练可以让模型在领域序列上继续学习。训练前要先区分问题是知识缺失、术语不熟、格式不对、检索缺失还是工具协议缺失。只有前两类更可能由继续预训练直接改善。

领域数据通常需要和一部分通用数据混合，以减轻灾难性遗忘。学习率、训练 token、数据重复、验证切片和通用能力回归都要记录。领域 validation loss 下降而通用代码或安全能力下降时，不能把训练称为成功。

### 6.15.2 SFT 的 loss mask 决定模型学什么

SFT 数据可以包含 system、user、assistant、tool call 和 tool result。若只希望模型学习 assistant response，labels 中的 system/user/tool 输入位置可以设为 ignore；若希望模型学习完整对话流或工具协议，目标 token 集合可能不同。

mask 不是越少越好。只训练极短的回答可能让模型忽略输入结构；把所有上下文都当 target 则可能浪费容量，甚至鼓励复述用户内容。目标应与部署时的生成边界一致，并用模板变体和多轮样例检查。

### 6.15.3 偏好学习的训练状态

偏好优化常需要 chosen、rejected、reference log probability 或 reward model 相关状态。数据比较的质量、长度偏差、拒答标签和 reference 版本都会影响结果。一个偏好 loss 下降，不代表事实性和任务能力一定提升。

后训练要持续保留基础能力、领域任务、安全和事实性评估。尤其是小数据偏好训练，可能在语气和格式上快速过拟合；训练步数、reference、KL 约束和数据混合应通过配对评估来分析。

## 6.16 训练系统的故障复盘：从指标异常到修复回归

下面用一个合成但完整的案例把前面的环节串起来。团队为“衡川”企业知识助手进行继续预训练，目标是让模型熟悉内部产品文档，同时保留通用代码能力。

### 6.16.1 初始设计

数据由产品文档 45%、通用网页 35%、代码 15%、操作手册 5% 组成。文档经过 OCR 和结构化解析，保留章节、表格标题和更新时间；每个样本带有来源 ID、版本和访问级别。训练使用固定 tokenizer，最大长度 8,192，文档按段落 packing，边界位置不允许跨文档 attention。

团队定义了三类验证：文档 token loss、代码 token loss、人工构造的产品问答和引用任务。系统监控有效 tokens/sec、padding 率、通信时间、gradient norm、checkpoint 写入时间以及每类验证指标。

### 6.16.2 观察到的异常

训练前 4,000 step 稳定，之后文档 loss 继续下降，代码 loss 开始上升，padding 率从 12% 变成 38%，有效 tokens/sec 下降 21%。总体 loss 仍然下降，因此如果只看单一曲线，问题不会立即暴露。

### 6.16.3 分层分析

首先，代码 loss 的退化与文档采样比例变化同时出现。数据统计发现文档解析后平均长度变短，packing 实现按样本槽位而不是有效 token 计算采样，导致文档实际 token 占比高于配置的 45%。

其次，padding 增加来自表格和标题被拆成许多短样本，packing 边界策略没有把同一 artifact 的相邻段落合并。最后，产品问答的引用正确率没有同步提高，说明领域 token loss 的改善还没有转化为证据选择能力。

### 6.16.4 修复和对照

修复分三步进行：

1. 采样器改为按有效 token 计权，并记录每个来源的实际 token 份额；
2. 在不跨权限和文档边界的前提下优化段落 packing，单独测 padding 率和 mask 正确性；
3. 增加引用任务的配对评估，区分知识记忆、检索选择和引用格式。

修复后先在小规模训练上对比旧实现和新实现，再恢复主训练。结果显示文档 loss 略高，但代码 loss 恢复，effective tokens/sec 提高 18%，引用支持率提高 9 个百分点。这里不能简单说“loss 变高所以模型变差”：目标数据分布、有效 token 权重和能力评估都发生了更合理的变化。

### 6.16.5 复盘结论

事故的根因不是某个单独的模型参数，而是采样单位、packing 利用率和评估目标之间没有保持一致。修复后新增的训练契约包括：数据比例按有效 token 统计，packing 记录边界 mask，所有 checkpoint 写入 sampler 版本，评估同时报告文档、代码和任务成功，任何数据配方变更都必须配套小规模对照。

## 6.17 用实验区分数据、配方和系统因素

### 6.17.1 先写可证伪假设

当某项能力下降时，不要立刻归因于“模型容量不够”。至少可以提出：

- 数据源比例变化导致目标 token 不足；
- tokenizer 或模板改变了监督粒度；
- 学习率、batch 或 warmup 改变了优化轨迹；
- 训练系统吞吐下降导致实际 token 预算不足；
- validation 或 benchmark 协议发生变化；
- 后训练或领域适配造成了能力遗忘。

每个假设都应对应一个可观察预测。例如，如果是数据比例问题，固定优化器和模型、只恢复旧 mixture 应能改善；如果是评估协议问题，固定 checkpoint 重新运行旧版和新版 evaluator 会显示分差来自测量；如果是数值问题，reference FP32 小跑应与低精度路径分离。

### 6.17.2 小规模实验矩阵

一个基本矩阵可以固定模型和 token 预算，只改变一个因素：

| 实验 | 数据 mixture | 学习率 | packing | 评估协议 | 要回答的问题 |
| --- | --- | --- | --- | --- | --- |
| A | 旧 | 旧 | 旧 | 固定 | 基线是否稳定 |
| B | 新 | 旧 | 旧 | 固定 | 数据改变的影响 |
| C | 旧 | 新 | 旧 | 固定 | 优化配方的影响 |
| D | 旧 | 旧 | 新 | 固定 | 系统实现是否改变语义 |
| E | 旧 | 旧 | 旧 | 新 | 评估代码是否造成差异 |

如果资源允许，再使用不同随机种子或重复小实验估计波动。单次小跑不能证明普遍因果，但能排除一批明显的实现和契约错误。

### 6.17.3 记录负结果

没有提升也是信息：增加某类数据没有改善目标能力，可能说明瓶颈在检索、任务格式或评估；更长上下文没有提升，可能是数据缺少长距离依赖或模型没有有效利用位置；更高 MFU 没有降低日历时间，可能是 checkpoint、通信或排队成为关键路径。

负结果要记录数据版本、训练 token、实际吞吐、评估切片和失败样例。否则下一轮会重复同一实验，只是换一个随机种子。

## 6.18 训练代码的可验证边界

### 6.18.1 参考实现与高性能实现

训练系统通常同时存在 Python 参考实现、框架算子、fused kernel 和分布式路径。参考实现速度慢，却更容易逐元素检查；高性能实现吞吐高，却可能在 mask、dtype、归约和边界条件上出现差异。

应使用相同输入比较：loss、per-token loss、梯度、参数更新和非有限值。对随机输入、极短序列、最大长度、全 padding、全 mask、不同 batch 和不同 dtype 做属性测试。高性能实现通过 benchmark 不等于语义正确。

### 6.18.2 一个训练统计的纯 Python 参考函数

下面的函数展示如何按有效 token 汇总多个 micro-batch 的损失，避免简单平均每批平均 loss。它不依赖 PyTorch，便于做测试：

~~~python
def weighted_loss_mean(batch_losses, batch_token_counts):
    if len(batch_losses) != len(batch_token_counts):
        raise ValueError("losses and token counts must have the same length")
    total_loss = 0.0
    total_tokens = 0
    for loss_value, token_count in zip(batch_losses, batch_token_counts):
        if token_count < 0:
            raise ValueError("token count must be non-negative")
        total_loss += float(loss_value) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("no valid tokens")
    return total_loss / total_tokens


assert abs(weighted_loss_mean([1.0, 3.0], [3, 1]) - 1.5) < 1e-12
assert abs(weighted_loss_mean([1.0, 3.0], [1, 1]) - 2.0) < 1e-12
~~~

第一条断言说明 token 加权平均为 `(3*1+1*3)/4=1.5`；第二条才是两个 micro-batch 等长时的普通平均。把这类小函数放进训练框架测试，可以提前发现 reduction 语义错误。

## 6.19 把训练结果写成可追溯结论

### 6.19.1 从“loss 下降”改成完整陈述

一句“loss 从 3.2 降到 2.8”缺少至少五个条件：在哪个数据切片，多少有效 token，使用什么 tokenizer 和 reduction，是否有污染，是否带来目标能力变化。更完整的结论应说明测量对象、对照、变化幅度和证据边界。

例如，合理的表述是：在固定 tokenizer、模板和 validation 版本下，继续预训练 20B 有效 token 后，文档切片的 token-level loss 从 3.20 降至 2.80；代码切片无显著变化，产品问答的引用支持率提高 6 个百分点，95% 区间和样本量见评估记录。这个表述仍然不证明所有知识都变好了，但读者知道结论覆盖到哪里。

### 6.19.2 训练报告的最小结构

一份可复核的训练报告可以包含：

1. 目标和基线：希望改变什么，和哪个 checkpoint 比。
2. 数据契约：来源、过滤、去重、采样和有效 token。
3. 模型与配方：结构、optimizer、batch、学习率、精度。
4. 系统配置：硬件、并行、吞吐、通信和故障。
5. 评估协议：validation、任务、人工、安全和污染检查。
6. 结果与副作用：切片指标、失败样例和资源变化。
7. 恢复与完整性：checkpoint、重启次数、轨迹连续性。
8. 结论边界：已证实、只在本设置观察到、仍需验证的部分。

这种结构不是写作模板，而是把训练过程的因果链保留下来。未来出现退化时，团队可以回到数据、代码、系统和评估证据，而不是重新猜测“这次到底改了什么”。

## 6.20 练习：把训练系统落到具体判断

### 练习一：算有效 token 和 padding

给出四条长度分别为 128、256、512、1,024 的样本，按最大长度 1,024 组成一个 batch。计算分配槽位、非 padding token 和 padding 浪费率；再假设其中 20% 的 token 是 prompt mask，计算 loss 有效 token。分别说明三个数字用于什么。

### 练习二：检查梯度累积归一化

两个 micro-batch 的有效 token 数为 100 和 300，平均 loss 分别为 1.0 和 2.0。计算 token 加权 loss 和简单批次平均 loss，并解释它们差异会如何改变梯度。

### 练习三：设计 loss spike 排查

给定“所有 rank 同时 spike”“只有一个 rank 先出现 NaN”“恢复后 LR 跳变”三个现象，分别列出最先检查的三类证据，并说明为什么不能只调小学习率。

### 练习四：比较并行方式

一个模型单层放不进单卡，但层间激活较小；另一模型每层都能放入单卡，但 global batch 需求很大。为两个场景分别选择 TP、PP、DP 或状态切分的组合，并写出通信代价和拓扑假设。

### 练习五：做一次训练配方消融

固定模型、总有效 token 和评估协议，只改变数据 mixture、学习率、packing 实现中的一个因素。为每个实验写出基线、预测、指标、失败样例和结论边界。

## 6.21 资料与证据边界

本章使用不同类型的公开资料支撑不同层次的论述：

1. Attention Is All You Need：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Improving Language Understanding by Generative Pre-Training：[https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
3. Training Compute-Optimal Large Language Models：[https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
4. Scaling Laws for Neural Language Models：[https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
5. Decoupled Weight Decay Regularization：[https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
6. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models：[https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
7. PyTorch DistributedDataParallel：[https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
8. PyTorch Fully Sharded Data Parallel：[https://docs.pytorch.org/docs/stable/fsdp.html](https://docs.pytorch.org/docs/stable/fsdp.html)
9. PyTorch Distributed Checkpoint：[https://docs.pytorch.org/docs/stable/distributed.checkpoint.html](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
10. PyTorch Automatic Mixed Precision：[https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html)
11. NVIDIA Mixed Precision Training：[https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
12. Megatron-LM：[https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
13. DeepSpeed：[https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
14. NCCL Documentation：[https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
15. FlashAttention：[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

原始论文用于支持模型目标、优化方法、scaling、ZeRO 和 attention 的方法或实验结论；PyTorch、NVIDIA、NCCL、Megatron-LM 和 DeepSpeed 资料用于支持 API、并行实现、混合精度和工程语义。公式中的显存、FLOPs、MFU、checkpoint 重算和单位成功成本是教学近似，实际值取决于模型结构、序列长度、硬件、框架版本、通信拓扑和数据处理。

论文中的 compute-optimal、scaling 和吞吐结果不能直接外推到任意模型或当前集群。一个公开框架的功能也不等于目标版本的实测性能。涉及具体模型、硬件或业务数据时，应以对应模型卡、版本文档、集群 profiling 和可复现实验为准。本轮写入前曾尝试联网复核资料入口，但当前环境 DNS 暂时无法解析外部域名；本章保留权威入口，正式发布前应在网络恢复后再次记录 HTTP 状态和版本。

## 6.22 结语：训练是一个可解释、可恢复的系统

训练系统的核心不是某个框架名，也不是某条 loss 曲线，而是能否把以下关系保持一致：

~~~text
目标能力决定数据和评估
    -> 数据契约决定有效 token 和采样分布
    -> tokenizer 与 packing 决定计算和监督粒度
    -> 模型、batch 与优化器决定更新轨迹
    -> 并行与精度决定显存、通信和吞吐
    -> 监控决定异常能否及时被看见
    -> checkpoint 决定故障后能否恢复
    -> 评估决定下一轮该改变什么
~~~

初学者应先能解释一次训练 step 中哪个 token 产生了梯度、一个 batch 为什么可能浪费计算、一个 checkpoint 为什么不只是参数文件，以及 loss spike 为什么需要证据链而不是一句“降低学习率”。有经验的工程师还要能把数据比例、有效 token、MFU、通信、故障重算和目标任务成功率放进同一个成本与能力账本。

当训练结果能够回答“学到了什么、用了多少有效数据、付出了多少系统成本、哪些副作用已经被测到、故障后如何证明没有改变语义”，训练才从一次昂贵的运行变成了可持续迭代的工程系统。下一章将把这套训练产物带入推理部署，讨论模型如何在有限显存、延迟和并发下提供稳定服务。
