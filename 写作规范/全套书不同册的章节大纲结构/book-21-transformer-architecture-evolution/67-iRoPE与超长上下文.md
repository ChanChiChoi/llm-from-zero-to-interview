# 第 67 章 iRoPE：局部 RoPE、全局 NoPE 与超长上下文

本章以 Llama 4 发布语境中的 iRoPE 为具体模型案例，解释 RoPE/NoPE 如何按层交错、局部与全局 causal mask 如何配对，以及 NoPE 路径上的推理期 query-temperature scaling 在什么条件下改变 attention logits。核心结论是：位置变换、可见范围、温度调节、频率缩放和长上下文训练彼此相关但不是同一个控制量。全章须区分 Meta 发布资料对 iRoPE 的用法、Scout 配置中的 scaled RoPE，以及后续研究文献较宽的术语解释；不把最大窗口或单一测试归因为某一个组件。RoPE 外推方法由第 18 章主讲，NoPE 顺序线索由第 64 章主讲，通用 hybrid 路径由第 65 章主讲，第 68 章负责 MetaP 与超参数迁移证据边界，通用训练稳定性由第五册主讲，缓存成本由第 69 章主讲，分层位置方案由第 72 章主讲。

## 67.1 iRoPE 的名称与公开口径

本节建立阅读 Llama 4 位置机制的模型与版本边界。Meta 在发布说明中把交错的无位置编码注意力层与推理期温度调节称为 iRoPE，并将 “i” 解释为 interleaved；发布期工程说明提供了更具体的每四层 NoPE 一层、RoPE 层 chunked attention 等结构信息。由于后续论文可能把 RoPE 频率缩放也纳入 iRoPE，正文应标明采用的是哪一来源的用法，而不是把缩写当作跨模型标准。

### 67.1.1 Meta 发布资料中的 iRoPE 命名

说明 Meta 将 iRoPE 描述为大多数层保留 RoPE，同时交错无位置编码 attention 层并在推理期做 attention temperature scaling；“i”指 interleaved。这个命名描述 Llama 4 的组合设计，不是一个有唯一数学定义、所有模型共用的 RoPE 算法名。后续引用缩写时须以来源原文确定其范围。

### 67.1.2 位置变换、可见范围与温度缩放是不同配置轴

区分 Q/K 是否应用 RoPE、attention mask 开放哪些 key、query 是否按位置缩放，以及 RoPE 频率是否重映射。Llama 4 发布期示例把 NoPE 与 global causal mask 配对、RoPE 与 chunked local causal mask 配对，并对 NoPE 层启用温度调节；这是一组具体配置，不意味着位置编码天然决定注意力窗口或局部/全局范围。

### 67.1.3 本章与相邻章节的分工

第 18 章解释 PI、NTK、YaRN 等一般 RoPE 扩展；第 64 章讲 NoPE 的顺序线索与长度泛化边界；第 65 章比较 local/global/recursive 历史路径；第 68 章讲长训数据与稳定性；第 69 章讲 cache 表示和增长；第 72 章讲其他分层位置尺度。第 67 章只提供理解 Llama 4 iRoPE 所需的最少回顾，重点放在具体交错结构、独立机制轴与公开证据。

## 67.2 RoPE 局部路径与 NoPE 全局路径怎样交错

本节把层表还原为 token 的直接可见路径。RoPE 变换给 Q/K 注入位置相位；chunk mask 再限制单层可读的前缀范围。NoPE 层不对 Q/K 施加 RoPE，但 Llama 4 配置可让它读取完整因果前缀。通过连续层之间的残差表示，远程内容可以进入某个 token 的表示并继续传递；这只是架构提供的通路，不保证检索、保留或回答正确。

### 67.2.1 RoPE 层的旋转与 chunked causal attention

简要唤回 Q/K 按位置旋转的作用，不重讲第 18 章的角频率和外推方法。随后区分 RoPE 与 local mask：发布期 Llama 4 结构中多数层使用 RoPE，且把 causal attention 限在当前 chunk，官方/框架示例的 chunk size 为 8192。窗口外 token 在该层不能直接影响当前 query；chunked mask 不等同于新的位置编码。

### 67.2.2 NoPE 层的全局因果读取

解释 NoPE 路径跳过 RoPE，而 global causal mask 允许当前位置直接计算与此前全部 key 的相关性、读取其 value。因果可见性仍保留顺序方向；若启用 temperature tuning，query 还会按位置乘以标量，因此“NoPE”准确指不做该层的 RoPE 变换，不应概括为所有计算都不含任何位置相关因子。详细的 NoPE 置换与隐式顺序性质回链第 64 章。

### 67.2.3 追踪一个跨 chunk 事实经过相邻层

用小型教学设置说明局部层和全局层怎样改变可达路径：RoPE 层的 query 只能读当前 chunk；后续 NoPE 全局层可以直接读远处因果前缀；更新后的 hidden state 再进入下一层。标明教学 chunk 尺寸不是 Llama 4 参数，且表示传递路径并不等于模型必然正确找到证据。

## 67.3 NoPE 路径上的推理期温度调节

本节解释按当前位置缩放 query 的机制和可声称范围。Softmax 的候选 key 随序列变长而增加，可能使注意力权重难以突出少数候选；Meta 参考实现选择在 NoPE 路径上随位置增长地放大 query，从而等价地放大 dot-product logits。须把论文提出的 Scalable-Softmax 动机、Meta 自定义实现和已验证的模型效果分开。

### 67.3.1 长前缀中的 softmax 竞争

从当前位置面对更多可见 key 出发，解释相同量级的相似度经 softmax 归一化后，少数候选得到的权重可能变小。Scalable-Softmax 原论文提供候选数增加导致权重分布变平的研究动机；该结果不是所有 attention head、数据分布或长上下文任务的统一定律，也不能由此直接认定 Llama 4 的 10M 能力原因。

### 67.3.2 位置依赖的 query scale 与 logit 变化

按 Meta 参考代码定义当前位置的缩放因子，说明其只在温度调节开关开启且当前层未使用 RoPE 时应用于 query。将 query 乘以因子如何使所有 QK logits 同比例放大写清楚，并解释对 softmax 分布的影响；符号 floor_scale、attn_scale 和零起始位置应与来源实现一致。具体 checkpoint 的启用状态和参数不得由库默认值替代。

### 67.3.3 锐化权重不是新增证据或长程能力保证

说明增大 logit scale 会提高分布集中度，但不会建立远程信息通路、补足长样本训练或保证聚焦到正确证据。过强缩放也可能让分布过尖；质量结论需要按模型 revision、长度和任务实测。Hugging Face 文章关于该因素可能促成 Scout 10M 的说法属于作者推测，不应写成组件因果结论。

## 67.4 Scout 的 RoPE 频率缩放与 iRoPE 术语差异

本节说明频率缩放在 Llama 4 中与层交错、mask 和 query-temperature tuning 的关系，并处理不同来源对 iRoPE 的命名差异。Meta 参考配置暴露 scaled RoPE 字段并注释其只用于 Scout；ICLR 2026 论文则用更宽口径把角频率缩放也归入 iRoPE。正文采用 Meta 发布语境作为主定义，同时保留该术语差异，避免由名称推断配置。

### 67.4.1 Scout 的 scaled RoPE 是可独立核验的配置

对照 Meta args.py 的 use_scaled_rope、rope_scaling_factor 与 rope_high_freq_factor，它们和 nope_layer_interval、attention_chunk_size、attn_temperature_tuning 分别存在。说明参考代码注释所指 Scout 特例及缺参回退值只约束该代码路径；比较模型时仍需读取 checkpoint 配置、代码 revision 和推理框架实际采用的设置。

### 67.4.2 后续论文对 iRoPE 的较宽描述

介绍 ICLR 2026 Frequency Entropy 论文把层间 RoPE/NoPE 与旋转角缩放并列描述为 iRoPE 的两项变化，并报告其对 Scout 4K 序列的 RoPE 维度分析。区分这是该论文的术语解释和分析设置，不等于 Meta 对命名的统一正式定义，也不支持把频率熵结果外推到 10M 的长任务能力。

## 67.5 Llama 4 的 10M context claim 能证明什么

本节把模型窗口声明、长上下文训练披露与具体任务评估分开。Meta 发布资料同时给出 Scout 256K 训练/后训练长度、专门数据集进行的 mid-training 长度扩展，以及 10M 输入窗口声明，并报告针尖检索与 10M code token 累积 NLL。公开信息没有提供足以重建完整长度课程的训练配方或将各组件单独消融的因果证据。

### 67.5.1 训练长度、mid-training 扩展与输入窗口

准确转述 Meta 对 Scout 256K 训练/后训练、specialized datasets 与 10M 输入长度的发布描述，避免将不同阶段简化成“只改了 RoPE 就从 256K 扩到 10M”。公开材料未披露完整样本配比、阶段顺序、所有优化超参数或必要组件消融；这些未知项应保留为未知。

### 67.5.2 针尖与长代码 NLL 的证据范围

针尖任务测量特定检索能力，长代码 NLL 反映特定连续文本上的语言建模表现；它们分别提供有限证据，不能单独覆盖多证据整合、冲突消解、长文档归因、缺失证据拒答或多模态推理。发布方报告的评测不可改写成独立复现，且模型整体结果不能归因到 iRoPE 单项。

### 67.5.3 任务级有效长度应按位置与证据结构分层

提出与本章机制相关的评估维度：长度、证据位置（开头/中间/末尾）、单证据与分散多证据、干扰和版本冲突、证据缺失、输出引用准确率及相应运行成本。将最大可接收 token 数、某项任务上的有效长度与部署服务长度分别报告；详细长上下文 benchmark 综述留给评测相关书册。

## 67.6 路径计算、缓存与失败边界

本节从结构解释注意力成本和容易误读的实现边界。局部 RoPE 层只有在执行后端利用 chunk mask 时，才会按局部访问量减少注意力工作；全局 NoPE 层仍允许访问完整前缀，因而仍有随长度增长的 dense attention 计算。局部可见范围也不自动裁掉旧 K/V。层路径只是长上下文系统的一部分，训练数据、cache 实现和硬件/运行时仍影响可用能力。

### 67.6.1 局部层与全局层的计算账本

在三比一交错的公开示例下，分别说明 chunked local attention 和 full-prefix global attention 的随长度计算趋势，并指出真正节省多少取决于 mask/kernel 实现。少数全局层仍可能主导超长输入成本；层比例本身不能推出端到端吞吐、显存或质量结论。通用混合路径成本比较回链第 65 章。

### 67.6.2 chunk mask 不等同于 KV eviction

使用 Meta 参考实现作为版本化例子：K/V 按 token 位置缓存；多 token 调用构造局部或全局 mask，而单 token 解码分支可能不传 mask，故应区分公开架构描述与具体代码路径。局部可见性限制本身不淘汰早期 K/V；淘汰、压缩或分页属于 cache/runtime 机制，显式 KV 与其他 cache 表示由第 69—71 章展开。

### 67.6.3 把位置、训练与运行时故障分开定位

将外推退化、长任务训练不足、NoPE 全局注意力成本、cache 增长和任务评测偏差作为不同问题观察，分别寻找对应证据，不把所有长输入失败都叫作 RoPE 问题。位置方法比较回链第 18、72 章，训练课程和短任务回归回链第 68 章；本章不扩写成通用 serving 运维清单。
