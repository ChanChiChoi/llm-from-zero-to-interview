# 第 74 章 CSA/HCA：压缩历史与 query 读取路径

本章围绕 DeepSeek-V4 技术报告中的 Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA），解释它们如何把 token-level KV 变成可直接参加 attention 的 learned compressed entries，以及每个 query 怎样读取这些条目。中心区别是：CSA 使用相对细的重叠压缩并经 Lightning Indexer 做 query-dependent top-k；HCA 使用更粗的非重叠压缩，并对因果上已经完成的压缩序列做 dense attention。两者都保留近期原始 KV 窗口，并让压缩项与局部项共同参与核心 attention。本章讲清存储轴、读取轴、因果完成条件、压缩误差和选择误差，不把模型级成本数字推广为所有稀疏/压缩注意力的效果。第 77 章负责 DeepSeek-V4 具体配置、百万 token 评测、低精度、KV cache 管理和参考实现；第 65 章负责混合路径总体比较，第 67 章负责位置编码与长上下文外推。

## 74.1 长上下文要同时控制历史存储与每步读取

本节从 token-level KV cache 的线性增长出发，拆分“保留多少历史表示”和“每个 query 访问多少条目”两个成本维度。通过与缩窄 KV 表示和稀疏选择相区分，明确 CSA 把 learned compression 与查询选择串成一个机制，HCA 则通过更激进压缩后密集读取。

### 74.1.1 传统 KV 逐 token 保存，改变宽度不等于缩短历史序列

KV cache 的历史长度会随已处理 token 数增长；GQA/MQA 减少每个 token 的 KV heads，MLA 减小每 token 表示宽度，但这些路线与沿序列维产生更少条目的 block compression 是不同的压缩轴。本节只唤回区分 CSA/HCA 所需的前置，不重讲第 69—70 章。

### 74.1.2 表示压缩与 query 级选择针对不同成本

压缩把多个 token 的表示映射成更少、可被后续 attention 直接读取的条目；选择根据当前 query 限制实际读哪些候选。前者影响长期保存的 state 数与信息粒度，后者影响 query 侧搜索和读取路径；CSA 同时包含两者，HCA 采用压缩而不做 indexer top-k。

### 74.1.3 CSA/HCA 是 DeepSeek-V4 中的具体混合注意力方案

本节把 CSA/HCA 限定为 DeepSeek-V4 技术报告中的正式模块名，统一符号 $n,m,m',k$，并说明后续机制描述对应报告 §2.3。它们不是任意模型里“压缩加稀疏”的通用缩写；章节不根据名称反推未公开的实现。

## 74.2 Learned pooling 怎样形成新的压缩 KV

本节说明两个模块共用的学习式 block-compression 思路：先生成每位置的 content 与 gate logits，再按块内位置对每个特征维分别归一化并加权合成。压缩结果本身成为后续 attention 的 key 和 value，不是指回原始 KV 的搜索索引，也没有逐 token 无损重建保证。

### 74.2.1 content 与 gate logits 分头投影

对输入 hidden state 序列分别施加内容投影与打分投影，打分还叠加可学习的块内位置偏置。两条投影分别决定“各位置提供什么特征”与“每个输出维从哪些位置取值”，区别于固定均值池化。

### 74.2.2 每个特征维在块内位置间学习加权

gate logits 沿块内位置维 softmax，而不同特征维拥有不同的位置权重。其结果可以在一个压缩向量内部保留来自不同 token 位置的不同成分；应将这种 coordinate-wise pooling 讲成真实张量关系，而不误称为对整个 token 做单一权重选择。

### 74.2.3 压缩条目本身参与 attention，并非原始 KV 索引

生成的向量作为持久的 compressed KV entry，被核心 attention 当作 key/value 读取；这不同于先用低成本摘要查出候选，再回到原文 token KV 做精确 attention。存储和细节保真的改变必须与检索索引分开讨论。

## 74.3 CSA：重叠压缩后选择少量全局条目

本节沿 CSA 的数据流解释两阶段机制：先用重叠的双投影池化形成相对细粒度的压缩历史和独立的 indexer-key 历史，再由当前 query 生成分数并选择 top-$k$ 压缩条目交给主 attention。压缩与选择应分别追踪，避免把 indexer 分数读成 token 重要性的真值。

### 74.3.1 a/b 两条投影令压缩窗口跨越相邻块

CSA 的 $a$ stream 对当前 $m$ 个位置投影，$b$ stream 对前一个 $m$ 位置块投影，两套 content 与 gate 参数分别参与位置加权。第一个压缩条目没有前块时用零权重 padding；这属于报告定义的数据流，不能未经证据添加设计动机。

### 74.3.2 Overlap 以 $m$ 为 stride，条目数仍约为 $n/m$

一个压缩条目可以使用当前块和前块、覆盖至多 $2m$ 个位置，但连续输出每次只前进 $m$，相邻条目会重叠一个块。因此压缩后的序列长度约为原长度的 $1/m$，不能把感受范围 $2m$ 错当成压缩倍率 $2m$。

### 74.3.3 Lightning Indexer 使用独立压缩 key 对候选条目打分

indexer 为压缩 key 建立自己的投影与 learned compressed-key memory，并从当前 query latent 产生较低维的 indexer query heads；各头经 ReLU 相似分数和输入依赖权重合并为条目分数。应区分 indexer memory 与主 attention 的 compressed KV 表示，不能将搜索表示与读取表示混作同一张量。

### 74.3.4 top-k 选压缩条目，不等于选出正确证据

每个 query 只对因果可用的压缩条目做 top-$k$，再把被选条目送入主 attention。top-$k$ 结果是模型内部的稀疏读取决策，不是可解释的 gold-token 标注，也不保证压缩表示中保留了查询所需的细节。

## 74.4 HCA：粗粒度压缩后密集读取压缩序列

本节说明 HCA 复用 feature-wise learned pooling 的基本形式，但改为更大的 compression ratio 和不重叠窗口；它取消 CSA 的 Lightning Indexer，直接密集读取全部已完成的压缩条目。HCA 的 dense 属性只针对压缩后较短的历史序列。

### 74.4.1 单 stream 与非重叠块形成每 $m'$ 一个条目

HCA 由一组 content/gate 投影产生单条压缩流，对互不重叠的 $m'$ 个原始位置加位置偏置后做逐特征维加权池化。序列长约为 $n/m'$；HCA 不使用 CSA 的相邻块 overlap。

### 74.4.2 Dense 作用在可用压缩条目上

query 对所有已完成且因果可见的 HCA compressed entries 执行主 attention，不经过 Lightning Indexer 或 top-$k$。它并未恢复对原始 token 逐项读取；密集读取的对象已经是压缩表示。

### 74.4.3 与 CSA 的差异在压缩尺度和选择阶段

CSA 保留较细的全局表示并用 top-$k$ 限制主读取集合；HCA 使用更粗粒度表示来降低条目数量，使得对全部压缩条目的读取仍可行。具体 $m,m',k$ 数值因模型配置而异，留到第 77 章按固定模型版本讨论。

## 74.5 长程压缩与近期原文共同进入 attention

本节把压缩全局路径与近期滑动窗口合起来，解释因果 mask 和共享 KV 的作用。核心要点是完整压缩块与当前 query 的可见关系，以及压缩条目和局部 raw entries 怎样共同成为 attention 候选；位置处理和 cache 的具体实现另章负责。

### 74.5.1 未完成压缩块只有在其完成后才可见

压缩条目要等组成它的块完整后才能作为一个整体被 query 读取，因而不会包含 query 后面的未来 token。边界位置、已完成条目计数和未完成尾部需与因果索引一致；不得用未来信息构造压缩条目再回填给较早 query。

### 74.5.2 局部窗口提供原始近期 token 和块边界内因果访问

两种长程压缩路径都保留近期未压缩 KV。窗口一方面保留近邻 token 的细节，另一方面让尚未形成完整压缩条目的近期信息按 token 级因果关系可见；它是模型明确加入的局部 attention 路径，不是由压缩摘要自动恢复的能力。

### 74.5.3 Shared-KV MQA 让各 query heads 读取同一压缩 entry

核心 attention 用多个 query heads 读取共享的单份 compressed KV entry，且该 entry 同时承担 key/value。须将 query-head 数量与 KV entry 共享方式区分；此处的 MQA 结构会影响存储账本，但不改变 CSA/HCA 压缩与选择的定义。

### 74.5.4 local 与 compressed 候选共用核心 attention 归一化

局部原始 KV 与 CSA 选中的条目或 HCA 全部已完成条目一并提供给 core attention；它们属于同一个 query 的候选集合和归一化过程。不能画成各自独立 softmax 后再加权的两支，除非讨论的是另一种明确结构。

## 74.6 CSA/HCA 混合路径的成本与误差边界

本节将 entry count、query 读取、indexer 搜索、表征损失和模型实测结果分开。通过符号账本说明机制能推出什么规模关系，再用来源边界限制具体性能结论；DeepSeek-V4 的层配置、低精度和百万上下文结果交由第 77 章展开。

### 74.6.1 存储条目数、主读取数与 indexer 搜索不是同一成本

忽略边界项，CSA 主压缩历史约存 $n/m$ 个条目，每 query 的 core attention 约读不超过 $k$ 个全局条目加局部窗口；indexer 仍需给全部因果候选压缩 key 打分。HCA 约存 $n/m'$ 个全局条目，并密集读取这些可用条目加局部窗口。宽度、低精度、kernel 与实现会改变字节和时间，符号条目数不能代替 profiling。

### 74.6.2 CSA 面临压缩损失与 top-k 漏选，HCA 面临更粗的表示损失

CSA 的 query 可能遇到两类断点：压缩时混合掉关键差异，或 indexer 没选到含相关信息的条目；HCA 不做稀疏选择，但把更长历史投进更粗的摘要，仍可能丢失 query 需要的细节。局部窗口只保障最近部分的原始粒度，不解决远程压缩误差。

### 74.6.3 压缩倍率、召回和延迟需在对齐条件下测量

改变 $m,m',k$ 会同时改变条目数、压缩粒度和 query 工作；要比较时需固定模型、任务、训练与实现条件，并分别测压缩表征影响、indexer 召回、实际主 attention 读取和资源指标。DeepSeek-V4 报告的模型级 FLOPs/KV 比率不是任意 CSA/HCA 配置的保证。

### 74.6.4 百万 token 主张属于完整 V4 配置，详数由第 77 章承担

论文的百万上下文结果和具体压缩倍率、层类型、局部窗口、低精度与 serving 实现共同相关；不能只归因于 CSA 或 HCA 的单一算子。本节只建立机制与成本坐标，第 77 章按 V4 Flash/Pro 的公开版本与 baseline 讲具体数字和实现证据。
