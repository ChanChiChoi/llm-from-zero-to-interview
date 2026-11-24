# 第 67 章 iRoPE：局部 RoPE、全局 NoPE 与超长上下文

读到 Llama 4 的 iRoPE，先把四个可分别配置的问题拆开：Q/K 是否接受 RoPE 变换、每层可以读取哪些历史 token、query 是否按位置缩放、RoPE 频率是否重映射。Llama 4 把其中几项组合使用，但它们各自改变的机制不同。

本章把 iRoPE 放回它最明确的公开语境——Llama 4。Meta 发布说明称，Scout 的大多数层使用 RoPE，模型交错了不使用位置编码的注意力层，并在推理期调整注意力温度；Meta 把名称中的 “i” 解释为 interleaved。[Meta 发布说明](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)；发布当天的 [Hugging Face 工程说明](https://huggingface.co/blog/llama4-release)进一步展示了三层局部 RoPE attention 与一层全局 NoPE attention 的结构。

这种组合为长距离信息提供不同的读取路径。它本身不说明模型在任意超长任务中都能找到、整合并引用远处证据；实际能力还取决于训练数据、具体权重配置、执行后端和评测任务。

## 67.1 iRoPE 的名称与公开口径

### 67.1.1 Meta 发布资料中的 iRoPE 命名

Meta 在发布说明中把 RoPE 层与交错的 NoPE 层，以及推理期 attention temperature scaling 放在 iRoPE 的介绍中。Hugging Face 的发布期说明同样将交错 NoPE 和温度缩放统称为 iRoPE。这个名称描述 Llama 4 的组合方案；不同来源对术语的范围仍有差别，引用时应回到来源原文。

发布期工程说明还披露一个具体的层安排：每四层中有一层使用 NoPE，另外三层使用 RoPE。它的注意力范围也与位置变换交叉组合：RoPE 层使用 chunked causal attention，NoPE 层使用全前缀 causal attention。读者可以把这些事实理解为一组公开配置，而不是由“iRoPE”三个字必然推出的公式。

### 67.1.2 位置变换、可见范围与温度缩放是不同配置轴

注意力层至少包含下面几项相互独立的问题。Llama 4 将其中若干项配对使用：

| 配置轴 | 它改变什么 | Llama 4 发布期示例 |
| --- | --- | --- |
| Q/K 位置变换 | query 和 key 是否按 token 位置旋转 | RoPE 层应用 RoPE；NoPE 层跳过该旋转 |
| 注意力可见范围 | 当前 query 可以读取哪些历史 key/value | RoPE 层使用分块因果 mask；NoPE 层使用全前缀因果 mask |
| query 温度调节 | attention logits 随当前位置怎样缩放 | 配置启用时，对 NoPE 层的 query 按位置乘以系数 |
| RoPE 频率缩放 | 各频段的旋转角怎样随位置变化 | Meta 参考实现单独暴露配置字段，注释指出 Scout 使用 scaled RoPE |

RoPE 决定位置如何进入 Q/K；mask 决定本层能访问哪些 token；query scale 改变当前 query 与可见 key 的打分幅度。频率缩放改变 RoPE 的旋转频率，和层间交错也可以分别核查。理解这些轴，能避免把“无 RoPE”“全局 attention”“超长外推”当作同一件事。

### 67.1.3 本章与相邻章节的分工

[第 18 章](18-RoPE与长上下文位置外推.md)讲 Position Interpolation、NTK-aware、YaRN 等一般 RoPE 外推方法；[第 64 章](64-nope与隐式顺序.md)讲 NoPE 路径的顺序线索和长度泛化边界；[第 65 章](65-hybrid-attention局部全局与递归状态.md)比较局部、全局与递归状态等通用历史路径；[第 68 章](68-metap与超参数迁移.md)讲 MetaP 的公开主张、超参数迁移及其与长上下文扩展之间的证据边界；训练配方与稳定性由第五册主讲；第 69—71 章分别展开显式 KV、latent cache 与递归状态缓存；[第 72 章](72-prope与局部全局位置.md)讨论其他分层位置策略。

本章只回顾理解 Llama 4 路径所需的基础，重点放在 NoPE/局部 RoPE 怎样交错、temperature tuning 怎样作用于特定路径，以及这些组件与公开长上下文主张之间的证据边界。

## 67.2 RoPE 局部路径与 NoPE 全局路径怎样交错

### 67.2.1 RoPE 层的旋转与 chunked causal attention

RoPE 在生成 attention logits 之前，按 token 位置旋转 query 和 key。第 18 章已推导旋转相位如何使点积包含相对位置关系，本章只需记住：RoPE 作用于 Q/K 的位置表达，不直接决定这一层能访问多远的 key。

Llama 4 发布期示例把 RoPE 与分块注意力配在一起。在分块 mask 生效的调用路径中，每个 token 只能读取自己所在 chunk 中、不晚于自己的 token；落在其他 chunk 的 key 在该层对它不可见。发布期工程说明给出的 chunk size 是 $8192$ token。具体实现是否在所有调用阶段都应用该 mask，还要按 runtime 路径核对。

### 67.2.2 NoPE 层的全局因果读取

NoPE 层不对 Q/K 施加 RoPE。在多 token 的调用中，Llama 4 参考实现为 NoPE 层选择 global causal mask，当前位置可以与此前全部 token 的 key 计算相关性，再从相应 value 中读取内容。逐 token 解码时缓存只包含当前位置及此前位置，因此也没有未来 key 可读；该代码路径的 mask 细节见 67.6.2。

因此，Llama 4 的“全局”描述的是该层的可见范围；“NoPE”描述该层 Q/K 跳过 RoPE。两者在发布期配置中配对，但承担不同职责。开启温度调节时，query 还会得到位置相关的尺度因子，所以下面的 NoPE 计算并非在任何情况下都完全没有显式位置依赖。

### 67.2.3 追踪一个跨 chunk 事实经过相邻层

用一个教学设置观察架构路径，并假设该层确实应用局部 chunk mask。把 token 分成长度为 $4$ 的 chunk，分别是 $0$–$3$ 和 $4$–$7$；假设 query 位于位置 $6$，相关事实位于位置 $1$。这里的 chunk size 只是演示参数，Llama 4 发布期公开值为 $8192$；具体服务实现的逐 token 解码路径可能采用不同 mask，不能从这张示意表推出。

| 经过的层 | query $6$ 能读取的位置 | 事实位置 $1$ 是否能直接进入该层 |
| --- | --- | --- |
| RoPE 局部层 | $4,5,6$ | 否，位置 $1$ 在另一个 chunk |
| NoPE 全局层 | $0$ 至 $6$ | 能，因果前缀包含位置 $1$ |
| 后续 RoPE 局部层 | $4,5,6$ | 不能重新读取位置 $1$，但位置 $6$ 的当前表示已包含前一层计算出的结果 |

在全局层，位置 $6$ 的 query 可以从位置 $1$ 的 value 读取信息，attention 输出再经残差路径进入下一层。之后局部层处理的是更新后的 token 表示。这展示了远程信息可能怎样跨过局部边界；模型仍可能给远程内容分配低权重、在层间改写掉它，或没能用它回答问题。结构提供可达路径，任务成功需要单独评估。

## 67.3 NoPE 路径上的推理期温度调节

### 67.3.1 长前缀中的 softmax 竞争

当 query 可见的 key 数量增多时，softmax 要在更多候选之间分配概率。Nakanishi 的 [Scalable-Softmax 研究](https://arxiv.org/abs/2501.19399)分析了候选数增长时注意力分布可能变平的问题，并提出一套替代 softmax 的方法。这篇论文提供温度缩放设计的研究动机；具体头、数据和任务的表现仍须单独验证。

Meta 的参考实现代码注释引用了这项研究，但 Llama 4 使用的是按当前位置定制的 query scaling 路径。理解动机时可以借鉴“在长前缀中保持打分差异”的问题，描述具体实现时则应以模型代码和 checkpoint 配置为准。

### 67.3.2 位置依赖的 query scale 与 logit 变化

Meta 参考实现为当前位置 $m$ 计算因子。下面用 $a$ 表示代码中的 `attn_scale`，用 $s$ 表示 `floor_scale`；位置编号从 $0$ 开始。令 $q_m$ 是位置 $m$ 的 query 向量，$k_j$ 是位置 $j$ 的 key 向量，$d_k$ 是 key 向量维度，$z_{m,j}$ 表示缩放前的点积 logit：

```math
\begin{aligned}
g(m) &= 1 + a \log\left(\left\lfloor \frac{m+1}{s} \right\rfloor + 1\right), \\
\widetilde{q}_m &= g(m)q_m, \\
z_{m,j} &= \frac{q_m^\mathsf{T}k_j}{\sqrt{d_k}}, \qquad
\widetilde{z}_{m,j}=g(m)z_{m,j}.
\end{aligned}
```

分段取整使调节在短位置范围内保持为 $1$，跨过 `floor_scale` 后才逐段增加。这里的 $\log$ 是自然对数。当 $m+1<s$ 时，$g(m)=1$；达到第一个边界时，因子变为 $1+a\log 2$。query 乘以 $g(m)$ 后，所有可见 key 对应的 dot-product logit 同比例放大，softmax 分布随之更集中。工程资料把这条路径称为 temperature tuning；从 logits 角度看，它提高了逆温度。

### 67.3.3 锐化权重不是新增证据或长程能力保证

位置缩放会改变当前 query 对可见 key 的权重分布，却不会增加可见的 token，也不会补上缺少的跨段训练样本。放大 logits 后，模型更容易集中于高分 key；若高分候选来自干扰或训练偏差，集中度本身不会让证据变正确。

因此需要把位置缩放和评测结果绑在具体模型 revision、配置、长度与任务上。Hugging Face 发布说明把温度调节称为可能促成 Scout 10M context 的因素，原文措辞是推测；公开材料没有提供单独关闭这项机制、保持其余设置不变的完整消融。章节只报告这项推测的来源，不把它写成已证实因果关系。

## 67.4 Scout 的 RoPE 频率缩放与 iRoPE 术语差异

### 67.4.1 Scout 的 scaled RoPE 是可独立核验的配置

Meta 当前参考实现的 `args.py` 把 `use_scaled_rope`、`rope_scaling_factor`、`rope_high_freq_factor` 与 `nope_layer_interval`、`attention_chunk_size`、`attn_temperature_tuning` 分别设置。源码注释说明 Scout 使用 scaled RoPE；验证代码在启用时会为缺失的缩放参数填入回退值。回退值只说明该实现如何补缺，实际 checkpoint 和 runtime 配置仍应单独读取。

Meta 参考代码的 [`model.py`](https://github.com/meta-llama/llama-models/blob/main/models/llama4/model.py) 还展示了频率缩放怎样改变不同波段的 RoPE 频率，而不是把整段 position id 简单除以一个常数。读者已在[第 18 章](18-RoPE与长上下文位置外推.md)学习通用频率扩展方法；这里保留 Scout 是一个模型配置案例。参数定义见可变的 [Meta main 分支 `args.py`](https://github.com/meta-llama/llama-models/blob/main/models/llama4/args.py)，结论绑定 2026-11-24 查阅的代码状态。

### 67.4.2 后续论文对 iRoPE 的较宽描述

ICLR 2026 的 [Frequency Entropy 论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html)把 iRoPE 描述为两项组合：层间交错 RoPE/NoPE，以及按比例缩放 RoPE 旋转角。论文在 Llama 4 Scout 上计算频率熵，使用 WikiText-103 构造 4,096-token 输入，并分析 RoPE 维度在模型表示中的使用。

这是一种较宽的文献用法。Meta 发布页把 iRoPE 的突出定义放在 interleaved 层和推理期温度缩放上；其参考实现又将 Scout frequency scaling 暴露为单独配置轴。正文以 Meta 的发布口径说明命名，同时介绍研究文献采用的较宽解释。频率熵分析不等于 10M 长任务评测，也不能单独证明 scaled RoPE 带来整个模型的上下文能力。

## 67.5 Llama 4 的 10M context claim 能证明什么

### 67.5.1 训练长度、mid-training 扩展与输入窗口

Meta 发布说明对 Scout 作了几项披露：其预训练和后训练长度为 256K；mid-training 使用专门数据集开展长上下文扩展；发布时宣称输入 context 可达 10M。发布资料没有完整给出这些阶段的长度课程、数据比例或所有训练配置，章节按原文分别陈述，不自行补出未公开的阶段细节。

这些披露说明超长窗口与中期训练扩展共同出现。读者不应把窗口扩展只归因于一个位置公式：本章讨论的位置表示与 attention 路径只是架构侧的信息，训练样本和评测设置也会影响长输入任务的结果。[第 68 章](68-metap与超参数迁移.md)进一步区分 MetaP 的超参数迁移主张与专门 mid-training 的长上下文披露；训练配方与稳定性排障见第五册[第 5 章](../../book-05-llm-training/chapters/05-预训练目标与训练循环.md)和[第 9 章](../../book-05-llm-training/chapters/09-训练稳定性与debug.md)。

### 67.5.2 针尖与长代码 NLL 的证据范围

Meta 发布页提到针尖检索结果，以及在 10M token 代码上累计 negative log-likelihood 的测量。这两类结果覆盖的是具体检索和语言建模视角。针尖实验没有自动覆盖跨段多证据合并、版本冲突和缺失证据拒答；代码 NLL 也不能单独回答模型能否依据超长仓库定位并正确整合多个文件中的证据。

这些是 Meta 报告的结果，不是本地复现。公开页面没有给出足以把 10M 结果因果归到 iRoPE 某一组件的完整对照。产品窗口代表模型/服务宣称可接受的输入长度；任务级有效上下文还要看问题类型和证据结构。

### 67.5.3 任务级有效长度应按位置与证据结构分层

要判断某一长度对某项任务是否有效，至少同时记录：

| 维度 | 需要改变或观察的条件 |
| --- | --- |
| 长度与位置 | 多个长度档；证据位于开头、中间、末尾，并避免固定位置捷径 |
| 证据结构 | 单条证据、分散多证据、冲突版本、证据缺失 |
| 输出质量 | 检索命中、答案正确、引用段落是否支持答案、无证据时是否拒答 |
| 资源条件 | 模型 revision、推理后端、输入/输出 token、并发与 prefill/decode 成本 |

对照时固定 tokenizer、模型权重、任务模板和输出预算，再比较位置或长度变化。报告应分别标明最大输入窗口、特定任务上的效果曲线和目标 runtime 中的可服务长度。一次针尖命中不能代替文档理解评估，单一最大长度也不能表达所有证据位置的成功率。

## 67.6 路径计算、缓存与失败边界

### 67.6.1 局部层与全局层的计算账本

若每四层中三层使用长度为 $w$ 的局部 chunk attention、一层读取完整的 $n$ token 前缀，且执行后端能利用局部结构，则一个四层组的 attention 工作量可粗略写成：

```math
C_{\mathrm{attn}} \sim 3\,O(nw d_{\mathrm{model}}) + O(n^2 d_{\mathrm{model}}).
```

这里 $d_{\mathrm{model}}$ 表示该层所有 attention heads 合计的投影宽度；表达式省略 Q/K/V 投影的细节、MLP、MoE、通信、mask 构建和 kernel 固定开销。它只用于看分块 mask 生效时随序列长度增长的趋势。全局层仍有 dense attention 的二次项，所以这组路径不等于整网线性复杂度。局部 mask 只有被具体 kernel 利用时，才会减少相应计算；单凭 mask 中有许多不可见格子，不能推出实测加速。Meta 当前 main 分支的单 token 调用还有一项 mask 边界，见下一小节。

vLLM 在 2025 年 4 月的发布期工程说明给出一组条件化示例：vLLM v0.8.3 起，8 张 H100 上 Scout 可服务 1M、Maverick 约 430K；8 张 H200 上分别报告 Scout 3.6M、Maverick 1M，并建议用多节点部署接近 Scout 10M。[vLLM 发布期说明](https://vllm.ai/blog/2025-04-05-llama4)展示了权重配置、backend、GPU 和并行方式会影响实际可服务长度。这些是特定日期和软件配置的报告值，不是当前通用上限。

### 67.6.2 chunk mask 不等同于 KV eviction

注意力 mask 控制某层本次计算读取哪些 key/value；cache 策略决定历史 K/V 是否保留、淘汰或压缩。Meta 参考实现按 token 位置写入 K/V，并把截至当前位置的缓存前缀交给 attention；当一次调用处理多个 token 时，局部或全局 mask 再限定可见范围。当前 main 分支只在序列长度大于 $1$ 时构造这两种 mask。序列长度为 $1$ 时两个 mask 都保持 `None`，block 会回退到同为 `None` 的全局 mask，attention 调用因此没有显式局部 mask。这个代码路径不能证明单 token 解码仍限制在局部 chunk；应检查目标 runtime 的实际实现和版本。

因此，结构图上“局部层”既不能直接推出 KV cache 按窗口截断，也不能保证每个逐 token 解码路径都沿用局部 mask。不同 serving engine 可能采用不同 mask、cache 布局或回收策略，需检查具体实现与运行配置。混合架构中的显式 KV 及其他 cache 表示由[第 69—71 章](69-显式kv-cache在混合架构中的角色.md)继续展开。

### 67.6.3 把位置、训练与运行时故障分开定位

超长输入退化可能来自不同层面：位置相位外推不稳、训练样本没有教会模型使用远处证据、global attention 对大量 key 分配不当、cache/runtime 对输入施加限制，或评测任务只测到了简单检索。定位时先按位置机制、长训数据、注意力路径和具体运行配置分别找证据，再决定要比较哪项组件。

[第 18 章](18-RoPE与长上下文位置外推.md)负责一般 RoPE 扩展比较；[第 68 章](68-metap与超参数迁移.md)负责 MetaP 与超参数迁移证据，并说明训练 token horizon、训练序列长度和推理上下文窗口的区别；通用训练稳定性由第五册第 9 章主讲；第 69—71 章分析 cache 表示与序列长度成本。本章关注的发布期 iRoPE 描述组合了局部和全局两类位置/信息路径；具体实现的调用边界、公开配置与评测证据共同决定可支持的结论范围。
