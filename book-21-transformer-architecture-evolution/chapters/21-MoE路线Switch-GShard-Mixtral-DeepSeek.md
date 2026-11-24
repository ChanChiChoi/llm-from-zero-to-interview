# 第 21 章 MoE 路线：Switch、GShard、Mixtral 与 DeepSeek-MoE

## 21.1 稀疏 MoE 把 FFN 容量变成条件路径

在 Transformer block 里，attention 先让每个 token 的表示吸收上下文；随后，逐位置的 FFN 对每个 token 做非线性变换。Dense FFN 对所有 token 使用同一组参数。MoE 改变的是这段 FFN 路径：它准备多个结构相同、参数不同的 expert，再让 router 按 token 的当前表示挑选其中一部分执行。

因此，这里的路由单位是某一层的 token 表示，不是整段 prompt，也不是用户请求。相同句子中的不同 token 可以走不同 expert；同一个 token 在更深一层也可以遇到不同 router。GShard、Switch 和 DeepSeekMoE 的论文都把 Transformer 的 FFN 位置作为稀疏 expert 层的落点；Mixtral 也把每层的 feedforward block 改成 expert 集合。[GShard](https://arxiv.org/html/2006.16668) [Switch Transformers](https://arxiv.org/html/2101.03961) [Mixtral](https://arxiv.org/html/2401.04088) [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/)

MoE 的设计意图是把总参数容量和每个 token 实际经过的路径分开：模型可以保存多个 expert，但某个 token 只计算被选中的部分。这个结构描述本身不说明总权重如何分布到设备、路由额外开销有多大，或训练和推理是否更快。第五册第四章已详细区分 total 与 active 参数，本册第二十二章再比较 Dense 与 MoE 的训练和服务代价。

“专家”也不是人先写好标签的数学模块、代码模块或语言模块。每个 expert 学到的是该层 hidden representation 上的一种变换；它是否形成稳定、可解释的领域分工，需要单独的模型分析证据。路由表上出现较多数学 token，只能说明这些 token 常被送到某些 expert，不能单凭频率断言该 expert 只“懂数学”。

## 21.2 跟踪一个 token 如何经过 router 与 expert

先把路由写成一条完整的数据路径。第 $l$ 层中，token $t$ 的 hidden state 记作 $h_{t}$；router 为该 token 产生各 expert 的选择分数，得到集合 $S_{t}$。被选中的 expert 各自计算同一个 $h_{t}$，之后模型按对应权重把这些输出合并：

~~~math
y_{t}=\sum_{e\in S_{t}}a_{t,e}E_{e}(h_{t}).
~~~

这里 $E_{e}$ 是 expert $e$ 的 FFN，$a_{t,e}$ 是该模型给 token $t$ 和 expert $e$ 的组合权重。权重如何产生、是否对已选 expert 重新归一化，属于路由实现的定义，不是所有 MoE 共享的公式约定；Switch、GShard 和 DeepSeekMoE 原文采用的门控口径并不完全相同。[Switch](https://arxiv.org/html/2101.03961) [GShard](https://arxiv.org/html/2006.16668) [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/)

实际执行时，系统会把选中同一 expert 的 token 表示排在一起，形成该 expert 的输入 batch。expert 算完后，系统利用保留的 token 身份和置换关系，把输出送回对应 token，再做加权合并。若 expert 分布在不同设备上，dispatch 与返回还要跨设备传输；具体通信原语和执行效率属于第二十二章的系统讨论。

下面的路由表是教学设定，只用来追踪对象，不代表真实模型的 router 输出；它忽略容量溢出和设备放置。

| token | 被选中的 experts |
| --- | --- |
| $t_{0}$ | $E_{0},E_{2}$ |
| $t_{1}$ | $E_{1},E_{2}$ |
| $t_{2}$ | $E_{0},E_{2}$ |

按 expert 打包后，$E_{0}$ 收到 $t_{0},t_{2}$，$E_{1}$ 收到 $t_{1}$，$E_{2}$ 收到 $t_{0},t_{1},t_{2}$。这时数据暂时按 expert 排列，但模型仍需保留每一行原来属于哪个 token；否则 expert 输出无法回到正确位置。expert 批处理、逆置换和按 token 合并连在一起，才构成 MoE 层的完整路径。课程讲义和工程笔记也用路由轨迹或 token-expert pair 展示了这层对象映射。[CMU 15-780 MoE 讲义](https://www.cs.cmu.edu/~aditirag/teaching/15780S26_files/Lecture19.pdf) [Fu-Yun Wang 的 MoE 路由笔记](https://g-u-n.github.io/blogs/mixture-of-experts.html)

## 21.3 Top-1、Top-2 与容量策略如何改变路由

最常见的一类路由是 token-choice：每个 token 按分数选择 top-$k$ 个 expert，其中 $k$ 表示每个 token 选择的数量。$k=1$ 时，token 只有一条 routed expert 路径；$k=2$ 时，同一 token 会把表示送到两个 expert，再合并两路输出。被选择的 expert 个数改变计算路径，但 gate 权重仍需按具体方法解释。

Switch 是 top-1 路由的代表。其论文先按全部 experts 计算 softmax 概率，再取最大概率对应的 expert；该 expert 的 gate 值会缩放输出。因此不能把它概括成“选中之后权重必定变成 1”。GShard 的实现则是特定 top-2 变体，包含第二路 dispatch 的容量处理规则。两者说明 top-1/top-2 只是选择数量，不能单独决定容量、归一化或溢出语义。[Switch §2.1–2.2](https://arxiv.org/html/2101.03961) [GShard §2.2](https://arxiv.org/html/2006.16668)

固定容量会给每个 expert 的 token batch 设上限。当 router 发来的 assignment 超过容量时，该 assignment 如何处理由方法和实现决定：可能被丢弃、改走其他路径，或用别的规则处理。Switch 论文描述了自己的 token overflow；不能据此说所有 MoE 都会用同一种 token 丢弃策略。容量公式、负载均衡损失、overflow 统计与 router 数值稳定由[第五册第四章](../../book-05-llm-training/chapters/04-模型架构与初始化.md)完整讲解，本章只保留容量会改变路由是否被执行这一结构事实。

另一个方向是 expert-choice：不由每个 token 挑固定数量的 expert，而由每个 expert 从 token 中挑自己的固定 bucket。这样一个 token 最终可能接收不同数量的 expert 路径。它与 token-choice top-$k$ 的分配方向和每 token 计算预算定义不同；原论文的结果绑定其 T5 规模、训练资源和任务，不应直接推广到当前所有 MoE 模型。[Expert Choice Routing](https://arxiv.org/html/2202.09368)

## 21.4 GShard、Switch 与 Mixtral 的路由落点

下面的表按论文披露的结构并列路线。它不是在相同模型、数据与设备上完成的控制实验，因此“后来出现”不等于“对所有任务更优”。

| 路线 | 论文中报告的结构 | 阅读时保留的范围 |
| --- | --- | --- |
| [GShard](https://arxiv.org/html/2006.16668) | 在所研究的 Transformer 中把隔层 FFN 替换为 position-wise MoE，使用一种 top-2 gating 变体；同时以 annotation API 和 XLA 编译器支持自动分片与并行执行 | 架构路由与 GShard 系统共同支撑了该论文的大规模多语翻译实验；规模、TPU 设置和质量结果不能作为所有 top-2 MoE 的通用表现 |
| [Switch Transformer](https://arxiv.org/html/2101.03961) | 把稀疏路由简化为每个 token 选择一个 expert，并用该 expert 的 gate 值缩放输出 | top-1 减少了每 token 的路由分支；论文的质量和效率结论仍限于其 T5 变体、数据与训练配置 |
| [Mixtral 8x7B](https://arxiv.org/html/2401.04088) | 每个 MoE 层包含八个 FFN expert，router 为每个 token 选择两个并组合它们的输出 | 这是 Mixtral 论文所报告的模型配置；expert 数和 top-2 机制不能推出其他 Mixtral 版本或部署系统的性能 |

GShard 同时是一条架构和分布式编译路线；Switch 的核心结构简化是 top-1；Mixtral 的论文披露则给出八个 FFN expert、每 token 选择两个。把它们并列的目的，是看清具体结构决策，而不是仅按总参数量或发布时间排出性能名次。Dense/MoE 的成本、资源拓扑和服务取舍留给[第二十二章](22-Dense与MoE训练服务trade-off.md)。

## 21.5 DeepSeekMoE：细粒度切分与共享路径

DeepSeekMoE 不是只把 routed expert 数量调大。论文提出两项不同的结构设计。

第一项是细粒度 expert segmentation：将较宽 FFN expert 的中间维度拆成更窄的 expert，并相应增加每个 token 激活的 expert 数量。在作者设定中，这样可以在保持总 expert 参数和计算口径不变时，提供更多被组合的子 expert。更多组合是结构上的可选性；它本身不证明这些子 expert 已形成互不重叠、人人可命名的知识区域。

第二项是 shared expert isolation：把一部分 expert 固定为 shared experts，让该 MoE 层中的每个 token 都经过它们；router 再为 token 选择其余 routed experts。论文的设计动机是让 shared path 承担跨上下文常见的信息，减少各 routed expert 分别学习共同内容的冗余。shared expert 被固定激活是结构事实；“它一定学到通用知识”则仍是需要按该模型验证的解释。

DeepSeekMoE 论文在固定总参数和激活参数的消融中报告了细粒度拆分与 shared expert 的结果；这些结果支撑作者在对应模型和实验条件下的观察，不代表相同收益会出现在任意任务或硬件上。[DeepSeekMoE，ACL 2024](https://aclanthology.org/2024.acl-long.70/)

## 21.6 专家分工的证据边界与章节边界

MoE 中的 expert 是学习到的函数路径，不是天然的“数学专家”“代码专家”或“中文专家”。课程讲义提醒，expert 处理的是该层 hidden features；特征分工不必对应人类熟悉的主题分类。[Stanford MS&E 319 MoE 讲义](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf) 因此，若论文声称专家专业化，应查看它如何定义、测量并消融这种分工，且把观察限定在对应模型和任务上。只看 router histogram、expert 编号或少量激活 token，不足以证明语义 specialization。

DeepSeek-V2 同时把 MLA 和 DeepSeekMoE 放进同一架构：MLA 改变 attention 的 key/value 表示与 cache，MoE 则改变 FFN 的 token 路由。联合系统的参数、训练或推理结果不能未经对照就全部归因给其中一个模块。[DeepSeek-V2](https://arxiv.org/html/2405.04434v5) MLA 的结构与 KV cache 由本册第四章和第十一章解释。

本章到此回答“token 如何选择并组合 FFN expert，以及不同路线具体改了什么”。训练阶段如何定义路由损失、容量和稳定化，请接第五册第四章；Dense 与 MoE 的预算、设备拓扑、训练/推理、尾延迟和成本比较，接本册第二十二章。
