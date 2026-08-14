# 第十章：Long Context、MoE、RAG、Agent 论文线

大模型的能力不只来自参数。更长的输入、更大的稀疏容量、可更新的外部知识、可执行的工具和环境反馈，都可以成为模型完成任务时使用的计算资源。本章讨论五条经常一起出现、但不应混为一谈的论文线：Long Context、Mixture-of-Experts、Retrieval-Augmented Generation、Tool Use 和 Agent Planning。

它们的共同问题是：当一次固定的 dense 前向传播无法满足任务时，能力应当从哪里来？长上下文扩大模型可以观察的输入范围，MoE 增加总参数而只激活部分专家，RAG 把知识库变成外部记忆，工具把计算和业务动作交给执行器，Agent 则把一次生成扩展为有状态的多步循环。

这些方向的代价也不同。窗口变长不等于模型会使用远处证据，专家总参数变大不等于每个 token 的计算变大，检索到文档不等于答案忠实，生成工具参数不等于动作授权，Agent 能写出计划不等于任务最终成功。阅读论文时，必须把能力来源、计算路径、评估协议和失败责任放在一起。

## 1. 先区分五种能力扩展

### 1.1 一个研究任务的五种实现

假设任务是写一份带来源的行业研究报告。纯参数模型依靠训练时存入参数的知识，可能知识过期或无法给出来源。Long Context 允许它在一次请求中读入长报告和历史材料。RAG 先从资料库中选出与问题相关的片段，再交给模型组织答案。Tool Use 让模型查询实时数据库、运行计算器或执行代码。Agent 则负责把检索、计算、核对和写作组织成多步过程。

MoE 处在另一层：它改变模型内部的容量和计算分配。一个 token 可能被路由到不同专家，从而让模型在相近的 active compute 下拥有更大的总参数空间。MoE 可以作为 Long Context、RAG 或 Agent 所使用的底层模型，但它本身不是外部记忆，也不是规划器。

### 1.2 机制、记忆和执行的边界

| 方向 | 改变的对象 | 能力来源 | 主要失败责任 |
| --- | --- | --- | --- |
| Long Context | 输入序列与注意力计算 | 更多当前请求中的 token | 位置外推、长程利用、干扰和成本 |
| MoE | 模型内部参数路径 | 稀疏激活的专家容量 | 路由、负载、通信和训练稳定性 |
| RAG | 外部知识链路 | 文档、索引和检索结果 | 召回、排序、证据冲突和引用 |
| Tool Use | 模型与执行器接口 | API、代码、数据库和环境 | 参数、权限、幂等、错误和副作用 |
| Agent | 有状态控制循环 | 计划、行动、观察和记忆 | 长程可靠性、循环、停止和恢复 |

Long Context 和 RAG 都可能把文本放进 prompt，但前者主要改变可容纳范围，后者主要改变证据选择和知识更新。Tool Use 和 Agent 都可能多次调用模型，但 Tool Use 关注一次调用的协议和执行，Agent 关注跨步骤的状态与目标。MoE 则是模型内部的条件计算机制。

### 1.3 先写清 claim 的形式

“支持百万上下文”“MoE 更强”“RAG 减少幻觉”“Agent 可以自主完成任务”都不是可直接验证的 claim。更可复查的写法是：

> 在固定模型、数据、硬件和评估协议下，某种机制是否在给定输入长度、检索预算、专家激活计算或工具调用预算内，提高目标任务成功率，同时满足延迟、成本和安全约束？

这个写法迫使研究者说明分母和预算。长上下文要报告输入长度和相关证据位置；MoE 要报告 active FLOPs、负载和通信；RAG 要报告检索候选和引用支持；Tool Use 要报告可用工具和权限；Agent 要报告最大步数、状态和失败恢复。

## 2. Long Context：能放进去不等于能用起来

### 2.1 上下文窗口的三个含义

“上下文长度”至少有三种含义：

1. **接口上限**：tokenizer、API 或 runtime 接受的最大 token 数。
2. **训练长度**：模型在训练中实际见过并被优化过的序列范围。
3. **有效使用长度**：模型能在该长度下找到证据、组合远距离信息并完成任务的范围。

接口上限只说明输入没有被拒绝。一个模型可以接受很长文本，却在中间位置忽略关键事实，或者因为无关内容过多而输出错误答案。论文和模型卡若只给最大 token 数，读者还需要寻找训练长度、位置编码方案、评估任务和成本。

### 2.2 Self-attention 的长度成本

对 hidden size 为 $d$、序列长度为 $n$ 的 dense self-attention，attention score 的形状大致是 $n\times n$。忽略常数和 head 划分，计算和中间存储可以写成教学近似：

~~~math

C_{attn}=O(n^2d),
\qquad
M_{score}=O(n^2).

~~~

长度从 $n$ 增加到 $2n$ 时，二次项大致变为四倍。真实成本还取决于 batch、head 数、精度、是否 causal、kernel、重计算和硬件带宽，但这个近似解释了为什么长上下文很快成为训练和推理瓶颈。

训练时还要保存反向传播需要的激活；推理时则要保存历史 KV cache。两者都使“只是把 max length 调大”不成为完整的长上下文方案。

### 2.3 近似 attention 与 exact attention

长上下文研究可以大致分成两类。一类改变注意力连接结构，例如局部窗口、块稀疏、低秩或线性 attention，试图把二次交互减少到更低复杂度。另一类保持数学上的 dense attention，但重新设计实现，减少显存读写和中间张量。

两类方法回答的问题不同。稀疏或线性结构改变了模型的归纳偏置，可能牺牲远距离全连接能力；IO-aware exact attention 不改变 attention 定义，主要改善硬件执行效率。论文不能只用“速度更快”比较它们，还要说明是否改变了计算图和近似误差。

### 2.4 FlashAttention：IO-aware 的精确注意力

标准实现可能显式生成 attention score 矩阵，再做 softmax 和矩阵乘法。这个中间矩阵需要在 GPU 的 HBM 与更快的片上 SRAM 之间读写。长序列下，内存搬运可能比算术操作更限制吞吐。

FlashAttention 的核心思想是 tiling 和 online softmax：把 Q、K、V 分块加载到片上存储，在块内完成必要计算和归一化，不把完整 $n\times n$ attention 矩阵写回 HBM。它仍然计算 exact attention，不是把 token 两两关系近似成稀疏图。

可以把它想成两种记账方式。普通实现反复把大张草稿纸搬到仓库，FlashAttention 尽量把小块材料留在工作台上边算边合并。收益来自减少 IO 和中间存储，而不是把理论上的二次交互改成线性。

FlashAttention-2 进一步调整并行切分、warp 工作分配和非矩阵乘法开销，使 kernel 更接近硬件矩阵乘法的利用方式。阅读这类系统论文时，要同时看 HBM 读写、峰值显存、序列长度、batch、精度和端到端 tokens/s，不能只看一个 kernel speedup。

### 2.5 PagedAttention：推理时的 KV cache 管理

自回归生成的第 $t$ 个 token 需要访问历史 token 的 key 和 value。若每次都重算历史，成本很高，因此 serving runtime 会保存 KV cache。对多层、多头和 batch 请求，KV cache 大致随历史 token 数增长：

~~~math

M_{KV}
\approx
2\,L\,n_{kv}\,d_{head}\,n_{token}\,b,

~~~

其中 $L$ 是层数，$n_{kv}$ 是 KV head 数，$d_{head}$ 是 head 维度，$n_{token}$ 是缓存 token 数，$b$ 是每个元素的字节数。GQA/MQA 可以降低 $n_{kv}$，但长上下文和并发仍会快速消耗显存。

传统连续分配会因为请求长度不同、结束时间不同和前缀共享产生碎片。PagedAttention 把逻辑连续的 KV 序列映射到非连续物理 block，借鉴虚拟内存的分页思想，让 allocator 更容易复用空间和共享前缀。

它解决的是 serving memory management，不是模型的长程理解。一个 PagedAttention 服务可以更高效地承载长请求，但如果模型没有在长长度上训练或评估，服务效率提升不会自动带来答案质量提升。

### 2.6 位置编码与长度外推

模型不仅要存储 token，还要知道 token 的相对或绝对位置。若训练序列长度为 $n_{train}$，推理时直接使用远大于它的 $n_{test}$，位置编码和注意力分布可能落入训练未覆盖的区域。

常见策略包括：

1. 直接用更长序列训练，让模型在目标位置范围内学习。
2. 对 RoPE 进行位置插值或 scaling，改变旋转频率范围。
3. 使用 ALiBi 等相对位置偏置，尝试改善长度外推。
4. 采用滑动窗口、全局 token、分段记忆或递归摘要。
5. 通过 RAG 先筛选证据，减少无关 token 的长度压力。

这些策略的机制不同。位置插值可能保留更多原有能力，但不等同于模型已经学会复杂长距离推理；滑动窗口降低计算，却限制任意远距离交互；递归摘要减少长度，却把原始证据压缩成了新的中间表示。评估必须匹配实际使用的长度和任务。

### 2.7 最大长度与有效长度

可定义一个带证据位置的任务成功率：

~~~math

A(n,p)
=P(\text{task success}
\mid\text{length}=n,\text{relevant position}=p).

~~~

如果只在相关事实位于开头或结尾时测试，得到的平均值可能掩盖中间位置的失败。更完整的评估要改变总长度、相关证据位置、干扰比例、证据数量和推理跨度。

对于多证据任务，还应测量模型是否能同时引用多个远距离片段，而不是只寻找一个“needle”。单点检索通过只能证明局部访问能力，不能证明长文档分析能力。

### 2.8 Lost in the Middle 与长上下文利用率

长上下文模型常见的一类现象是：相关信息位于上下文中间时，性能低于位于开头或结尾。它提示研究者区分“attention 可以访问”与“模型在任务中会利用”。无关文本、相互冲突的证据和不同段落之间的实体对应会进一步放大问题。

因此长上下文评估至少应包括：

1. 单事实定位。
2. 多事实合并。
3. 中间位置和随机位置。
4. 无关 token 和近似干扰。
5. 需要引用原文的答案。
6. 需要跨段推理或时间顺序判断的任务。

如果模型只在长文本末尾的答案位置表现好，可能是位置偏置或提示格式问题，而不是一般的长程理解。

### 2.9 长上下文的单位成本

长上下文的质量提升要与输入成本和延迟一起报告。一个请求的粗略账本可以写成：

~~~math

C_{request}
=C_{prefill}(n_{in})
+C_{decode}(n_{out})
+C_{KV}(n_{in},\text{batch})
+C_{storage}.

~~~

如果把完整知识库全部放入 prompt，输入 token 和 prefill latency 可能远高于 RAG 先筛选片段的方案。长上下文与 RAG 的比较应保持任务证据一致，并同时报告准确率、引用支持、输入 token、TTFT、吞吐和峰值显存。

### 2.10 复现长上下文论文

一个可控的复现路径是先使用合成文档，把唯一事实放在不同位置，并加入可控的干扰段。然后逐步增加：

1. 总 token 长度。
2. 相关事实数量。
3. 证据之间的推理距离。
4. 近似和冲突文档。
5. 输出引用要求。

每个长度点至少保存模型配置、位置编码、推理 kernel、精度、输入 token、输出 token、峰值显存和请求延迟。不能把“runtime 接受输入”当作理解能力的通过条件。

## 3. MoE：总容量与 active compute 的分离

### 3.1 Dense 与 sparse 的差异

Dense Transformer 的每个 token 都经过同一组 FFN 参数。扩大参数通常会增加每 token 计算。MoE 则把 FFN 替换或扩展为多个 experts，由 router 根据 hidden state 选择少数专家。

它的核心目标是 conditional computation：模型总参数可以增加，但一个 token 只激活一部分。对 MoE 账本，至少要分开总参数、共享参数、active parameters、理论 FLOPs、实际通信和存储容量。

### 3.2 Router 与 top-k 路由

设 token 的 hidden state 为 $h$，有 $E$ 个专家。一个简单 router 可以写成：

~~~math

p(h)=\operatorname{softmax}(W_rh),
\qquad
S(h)=\operatorname{TopK}(p(h),k).

~~~

输出可以表示为：

~~~math

h'
=\sum_{e\in S(h)}\tilde p_e(h)E_e(h),

~~~

其中 $E_e$ 是第 $e$ 个专家，$\tilde p_e$ 是选中专家分数重新归一化后的权重。top-1、top-2 和更复杂的路由在通信、容量、冗余和容错上不同。

初学者可以把 router 想成分诊台：先根据 token 特征分配给不同专科。专家不一定天然对应“法律”“数学”或“某种语言”，除非有分析证据证明这种 specialization；看到专家分工可视化不能直接推出因果解释。

### 3.3 Active parameters 与总参数

一个教学近似是：

~~~math

C_{MoE}
\approx
k\left(N_{shared}+\rho N_{expert}\right)D,
\qquad 0<\rho\leq1,

~~~

其中 $N_{expert}$ 是所有专家参数总和，$\rho$ 是平均激活比例，$D$ 是 token 数。这个公式没有计入 router、dispatch、all-to-all、padding 和负载不均衡，因此不能直接当作 GPU 账单。

MoE 可能拥有比 dense baseline 更多的总参数，却只有相近的 active FLOPs。它的收益来自更大的容量和条件路径，代价则转移到了参数存储、路由通信和并行调度。比较 MoE 与 dense 时，只报总参数或只报 active parameters 都不完整。

### 3.4 负载均衡

如果所有 token 都被送往少数专家，热点专家会成为瓶颈，其他专家闲置。设 $f_e$ 是路由到专家 $e$ 的 token 比例，$p_e$ 是 router 对专家 $e$ 的平均概率，一个常见的负载均衡思想是让两者都接近 $1/E$：

~~~math

L_{balance}
\propto
E\sum_{e=1}^{E}f_ep_e.

~~~

不同论文的辅助损失、系数和容量策略并不相同。公式的重点是提醒我们同时关注离散路由分配 $f_e$ 和连续 router 概率 $p_e$，而不是只看总 loss。

### 3.5 Capacity factor 与 dropped tokens

每个 expert 的处理容量通常有限。若一个 batch 有 $T$ 个 token、$E$ 个专家、每个 token 选 $k$ 个专家，可以用一个教学化容量近似：

~~~math

capacity
=\left\lceil
\gamma\frac{kT}{E}
\right\rceil,

~~~

其中 $\gamma$ 是 capacity factor。容量过小会造成 token 被丢弃、跳过或退化处理；容量过大则浪费显存并降低吞吐。论文应报告 dropped token ratio、每专家负载分布和容量设置，否则读者无法判断质量与系统成本的关系。

### 3.6 Sparsely-Gated MoE 与 Switch Transformer

Sparsely-Gated Mixture-of-Experts 奠定了用稀疏 gate 扩大模型容量的路线。Switch Transformer 进一步采用 top-1 routing，减少每个 token 的专家计算和通信，并配合负载均衡和容量管理扩展大模型训练。

top-1 的优点是路径简单、通信少，缺点是路由错误时没有第二个专家提供补偿。top-2 可能增加冗余和质量，但也增加 dispatch、通信和容量压力。论文比较不能只看模型参数，还要保持训练 token、active FLOPs、数据和硬件预算可比。

### 3.7 MoE 的专家分工能否直接解释

训练后观察到某些专家更常处理代码或某种语言，并不自动证明专家被明确训练成这些领域的模块。频率可能来自数据比例、tokenizer、router shortcut 或 batch 结构。要研究 specialization，可以做：

1. 按输入类型统计专家负载。
2. 固定输入做 router 干预或专家替换。
3. 比较去掉某专家前后的任务变化。
4. 在新领域和新语言上做 holdout。
5. 区分 router 选择相关性与专家输出的因果作用。

这使 MoE 论文从“看起来有分工”变成可检验的机制 claim。

### 3.8 MoE 的训练和部署成本

MoE 的总参数通常需要存储，即使每个 token 只激活少数专家。分布式训练还要把 token 发送到专家所在设备，all-to-all 的成本可能压过省下的算术计算。推理时 batch 组成、路由热点、专家并行拓扑和请求混合会影响实际吞吐。

一个更完整的 step 时间账本是：

~~~math

T_{step}
\approx
T_{compute}
+T_{dispatch}
+T_{all\text{-}to\text{-}all}
+T_{imbalance}
+T_{checkpoint}.

~~~

因此 MoE 论文必须同时报告 quality、active FLOPs、tokens/s、通信占比、专家负载、显存和故障恢复。单卡 FLOPs 降低不代表端到端成本下降。

## 4. RAG：把外部知识变成可追踪证据

### 4.1 参数记忆与非参数记忆

参数记忆的优点是调用快、表达紧凑、可以在生成中直接使用；缺点是更新需要训练或编辑，知识来源难以追踪，私有和新鲜资料不一定覆盖。RAG 引入非参数记忆，把文档或数据库内容保留在外部索引中，在请求时检索并注入上下文。

RAG 的目标不是把模型变成数据库，而是把模型的语言能力与外部证据连接起来。它可以改善时效性、领域覆盖和来源透明度，但引入新的失败环节：正确文档是否被召回，证据是否完整，模型是否忠实使用。

### 4.2 RAG 的基本数据流

一个典型系统包括：

~~~text
文档采集与版本管理
  -> 解析、切分、去重与权限标注
  -> embedding / keyword index
  -> query 改写与召回
  -> rerank / filter
  -> context packing / compression
  -> generator 生成带证据的回答
  -> citation / verifier / audit
~~~

每一步都可能决定最终结果。一个生成模型在没有正确文档的情况下回答错误，不应简单归因于“模型幻觉”；一个正确文档被召回但未被使用，才更接近上下文利用或生成忠实性问题。

### 4.3 召回与重排

稠密检索用 embedding 相似度寻找语义相关片段，BM25 等关键词检索更擅长精确实体、编号和术语。混合召回、query expansion 和 reranker 可以互补，但每多一层也增加延迟、版本和调参空间。

对一个问题集合，最基本的检索指标是 recall@k：

~~~math

Recall@k
=\frac{\#\{\text{queries with a relevant item in top-}k\}}
       {\#\{\text{queries with an annotated relevant item}\}}.

~~~

如果答案需要多个片段，还要定义 multi-hop recall 或 evidence set recall，不能只用“至少召回一段”掩盖缺失的第二个证据。

### 4.4 Chunking 是证据边界设计

切分不是一个只影响向量库大小的工程参数。chunk 太短，定义、条件和例外可能被拆开；chunk 太长，召回会带入无关内容，reranker 和 generator 的上下文压力增加。

合适的切分应尽量保留文档结构：标题、段落、表格、代码块、页码、时间和权限。对法律条款、代码函数和财报表格，按固定字符数切分往往会破坏语义单元。评估时要保存 chunk 与原文版本的血缘，以便回答引用能够回到确切来源。

### 4.5 RAG 与 Long Context 的关系

Long Context 主要解决“最多可以放多少”，RAG 主要解决“哪些证据值得放进去”。上下文窗口足够大时，把整个知识库塞入 prompt 仍然可能带来高 prefill 成本、干扰和中间位置利用问题。RAG 可以先筛选相关内容，再用长上下文容纳多个候选证据和历史对话。

反过来，RAG 的 top-k 不是越小越好。如果问题需要跨文档聚合，过度压缩会漏掉必要证据；如果模型无法处理长输入，召回太多又会降低利用率。两者应在相同证据覆盖下比较质量、延迟和 token 成本。

### 4.6 生成忠实性与引用支持

RAG 的回答可能看起来合理，却没有被检索内容支持。可以定义一个教学化的 claim support ratio：

~~~math

SupportRatio
=\frac{\#\{\text{answer claims supported by cited evidence}\}}
       {\#\{\text{verifiable answer claims}\}}.

~~~

这个指标需要 claim 抽取和证据标注，不能由答案是否包含引用链接代替。引用存在但与具体句子不匹配，仍然属于 citation misalignment。

还要区分“证据支持”与“答案正确”。文档本身可能过期或错误；模型忠实引用错误文档，仍然会产生错误答案。因此需要同时检查文档质量、时间、权限和冲突处理。

### 4.7 RAG 的典型失败链

| 现象 | 更可能的责任层 | 需要检查 |
| --- | --- | --- |
| 正确文档不在 top-k | 召回 | query、embedding、BM25、文档覆盖 |
| 证据被切成互不完整的片段 | 解析/切分 | 文档结构、overlap、表格和代码 |
| 相关文档在候选中但未进入 prompt | 重排/预算 | reranker、top-k、权限过滤、压缩 |
| 证据在 prompt 中但答案忽略 | 生成/利用 | 位置、冲突、指令、引用约束 |
| 答案引用了不支持的文档 | 生成/引用 | claim-evidence 对齐、verifier |
| 过期或越权文档被使用 | 治理/权限 | 时间版本、租户、ACL、删除血缘 |

这个表比笼统地说“RAG 有幻觉”更有用，因为每种失败需要不同修复。

### 4.8 RAG 论文的公平 baseline

一个可信的 RAG 实验至少应包含：不检索的 generator、简单关键词检索、稠密检索、rerank 版本和可能的 oracle evidence。所有版本要固定 generator、解码参数、上下文预算、知识库版本和评估 prompt。

如果 RAG 使用了更多输入 token、额外的 reranker 和更强的模型，却只与一个短 prompt baseline 比较，质量提升无法归因于检索本身。报告还要说明查询是否见过答案、索引是否包含评估集近重复和文档权限如何处理。

### 4.9 RAG 的小规模复现

一个教学闭环可以从几十篇有明确答案出处的文档开始：

1. 建立原文、段落 ID、版本和权限字段。
2. 实现 BM25 或简单关键词 baseline。
3. 加入 embedding retrieval。
4. 固定 top-k 后比较 reranker。
5. 记录 relevant recall、answer accuracy、support ratio 和 latency。
6. 对召回失败、使用失败和引用失败分别采样分析。

不要一开始就追求复杂向量数据库。若不能在小数据集上区分“没召回”和“召回但没用”，扩大系统规模只会把错误埋得更深。

## 5. Tool Use：模型输出变成受约束的动作

### 5.1 为什么需要工具

语言模型擅长把自然语言映射成文本，但不天然擅长精确算术、实时查询、私有数据库访问、代码执行和业务系统操作。工具把这些能力交给外部执行器，让模型负责选择、参数化和结果解释。

工具调用的价值不只是“知道更多”，还包括得到可验证的计算结果和环境反馈。代价是模型输出不再只是文本，错误可能改变外部状态，安全边界因此从回答内容延伸到权限和副作用。

### 5.2 工具调用的协议对象

一个工具调用至少有：

1. 工具名称和版本。
2. 输入 schema 与必填字段。
3. 用户、租户和执行器权限。
4. 超时、重试和幂等语义。
5. 返回值 schema、错误码和敏感字段处理。
6. trace ID、审批状态和状态差分。

模型生成一个正确的函数名只是协议链的第一步。参数类型错误、权限不足、过期状态、重复执行和结果误读都可能让一次调用失败。

### 5.3 Toolformer 与可学习的调用行为

Toolformer 等工作研究如何让模型学习在需要时插入 API 调用，并把调用结果重新放回上下文。其核心问题不是让模型记住工具说明，而是学会判断：什么时候调用、调用什么、参数是什么，以及结果如何帮助继续生成。

这类论文需要区分 API 选择准确率、参数正确率、结果使用率和端到端任务成功率。模型可能选对工具却传错日期，也可能参数正确却忽略返回值。只报告工具调用格式合法不够。

### 5.4 结构化输出与 schema 约束

function calling 或 JSON schema 可以把自然语言意图限制为结构化对象。约束解码能减少语法错误，但不保证语义正确或动作安全。

例如 `user_id` 字段满足字符串 schema，不代表模型选择了当前用户的 ID；`amount` 是合法数字，也不代表转账金额经过授权。schema validation、业务校验、权限检查和人工审批是不同层次。

### 5.5 工具结果如何进入上下文

工具返回结果通常会被序列化为新的 observation，再交给模型。系统要处理结果长度、字段可信度、错误信息、未授权数据和 prompt injection。工具返回的网页内容或文档也应被视作外部输入，不能自动成为系统指令。

一个稳健的数据流是：

~~~text
模型提出调用
  -> schema validation
  -> policy / permission check
  -> executor 在受控环境运行
  -> 结果清洗、标记来源和敏感字段
  -> 以 observation 回传模型
  -> 模型决定继续、澄清或结束
~~~

模型是否有权决定“继续调用”与执行器是否真正允许调用，应保持分离。自然语言中的自信不能提升权限。

### 5.6 工具使用评估

可以把工具任务拆成几个条件指标：

~~~math

P_{success}
=P_{select}
  \times P_{args\mid select}
  \times P_{exec\mid args}
  \times P_{use\mid result}
  \times P_{final\mid trace}.

~~~

这个乘法是教学化分解，不假设各项真正独立，但它说明端到端成功率下降的原因可能来自任何一段。应分别记录工具选择、参数、执行、结果使用、最终回答和状态副作用。

### 5.7 工具安全与幂等恢复

读操作、计算操作和写操作不应使用同一权限。写操作要有最小权限、参数范围、审批、幂等键、预览和回滚。重试必须知道请求是否已经执行，否则网络超时可能导致重复扣款、重复发信或重复修改。

工具研究论文若只在无副作用的模拟环境中报告成功率，结论不能直接外推到生产。需要说明环境是否可重置、工具是否真实执行、错误是否注入以及权限是否与现实一致。

## 6. Agent Planning：从单次生成到有状态循环

### 6.1 Agent 的最小状态机

一个 Agent 可以抽象成状态、目标、动作和观察的循环：

~~~math

s_{t+1}=F(s_t,a_t,o_{t+1}),
\qquad
a_t\sim\pi_{\theta}(\cdot\mid s_t,g),

~~~

其中 $s_t$ 是历史、计划、约束和环境状态，$g$ 是任务目标，$a_t$ 是工具或文本动作，$o_{t+1}$ 是环境返回。Agent 的难点在于状态不是静态 prompt，而是在每一步执行后改变。

### 6.2 ReAct：推理与行动交替

ReAct 把 reasoning trace 和 action 交替起来：模型先根据目标和当前观察决定下一步，再调用搜索、数据库或环境，随后读取 observation 更新状态。

~~~text
目标
  -> reasoning about next step
  -> action / tool call
  -> observation
  -> revised reasoning
  -> next action or final answer
~~~

它比只生成一条 chain-of-thought 多了环境反馈，也比只生成 action 多了对目标和中间结果的显式组织。ReAct 的论文价值需要结合具体环境：开放域检索、交互式任务和工具调用的反馈结构不同，不能把一个环境的成功率外推到所有 Agent。

### 6.3 Tree of Thoughts：让中间步骤进入搜索

线性推理在早期做错一步后，后续步骤可能围绕错误前提继续展开。Tree of Thoughts 允许生成多个中间 thought，对候选状态进行评估，再选择、扩展或回溯。

设每一步平均保留 $b$ 个分支，搜索深度为 $d$，未经剪枝的候选数近似为：

~~~math

N_{nodes}\approx\sum_{i=0}^{d}b^i.

~~~

这解释了 ToT 的代价：质量可能提高，但模型调用、评估和上下文复制迅速增加。若评估器本身不可靠，搜索会更有效率地放大错误判断。

### 6.4 反思、记忆与计划的区别

Agent 系统常把历史记录称为 memory，把重新评价失败称为 reflection，把未来步骤称为 plan。三者不应混为一个长文本：

| 对象 | 作用 | 主要风险 |
| --- | --- | --- |
| Plan | 描述尚未完成的目标和步骤 | 早期错误计划持续影响后续 |
| Working memory | 保留当前任务的证据和状态 | 上下文增长、信息冲突 |
| Long-term memory | 跨任务保存可复用事实 | 过期、隐私、错误记忆 |
| Reflection | 从失败中提取修复信号 | 语言上的反思不等于实际改进 |

记忆越多不代表 Agent 越强。错误、过期或越权记忆会污染新任务；写入长期记忆也应有来源、权限、时间和删除机制。

### 6.5 长程可靠性

若每一步独立成功概率都为 $p$，连续 $T$ 步全部成功的教学近似是：

~~~math

P(\text{all steps succeed})\approx p^T.

~~~

真实 Agent 的步骤并不独立，后一步可能纠正前一步，也可能放大前一步错误。但这个公式说明多步系统为什么需要恢复和检查点：即使单步成功率很高，长链条也会积累失败。

Agent 评估应记录平均和 P95 步数、每步工具错误率、恢复成功率、循环比例、停止原因、最终任务成功率和单位成功成本。

### 6.6 停止、恢复与人工接管

一个可部署 Agent 必须定义何时结束、何时重试、何时降级、何时询问用户以及何时交给人工。停止条件可以来自目标完成、预算耗尽、风险动作待审批、重复状态或环境不可用。

恢复策略不能只让模型“再试一次”。应区分参数错误、权限错误、瞬时网络错误、业务冲突和未知副作用，并保存执行前后的状态差分。否则 Agent 可能把一个不可逆动作重复多次。

## 7. 组合系统：边界交汇处最容易出错

### 7.1 Long Context + RAG

RAG 可以先过滤候选证据，Long Context 再容纳多份文档、历史对话和引用要求。组合时要处理排序、冲突、重复、权限和上下文预算。

一个长窗口可能让更多证据进入模型，却也让相关片段出现在中间位置；一个强 reranker 可能提高相关性，却误删解决多跳问题所需的第二片段。组合实验要记录 evidence recall、位置、context packing、回答支持率和成本。

### 7.2 RAG + Tool Use

RAG 通常读取版本化文档，Tool Use 可以查询动态数据库或执行计算。一个企业助手可能先检索制度条款，再查询当前用户状态，最后判断是否允许某个动作。

这里不能把文档中的“应该做什么”和数据库中的“当前状态”混成一个无来源文本。每个 observation 应标注来源、时间、权限和可信度，最终回答或动作要能回放到具体证据。

### 7.3 Tool Use + Agent

工具是动作接口，Agent 是跨动作控制器。Agent 的计划可以选择工具，但执行器仍应独立检查权限和参数。Agent 不能因为自己在计划中写了“用户已授权”就获得授权。

评估组合系统时，要同时看任务成功、工具成功、未授权调用、状态变化、恢复率、调用次数和成本。只看最终回答会忽略中间已经发生的危险动作。

### 7.4 MoE + Long Context

MoE 增加模型容量，Long Context 增加输入范围，组合后可能提高多领域长文档任务能力，但也会叠加 expert dispatch、attention IO、KV cache 和跨卡通信。长序列 token 分布还可能改变 router 负载，使训练或服务出现新的热点。

因此组合论文需要拆分贡献：dense/MoE 对照、短/长上下文对照、active FLOPs、专家负载、位置分桶和端到端成本都不能省略。

## 8. 如何评价这类论文的贡献

### 8.1 先判断论文层级

Long Context 论文可能是位置机制、训练配方、attention 结构或 kernel 系统；MoE 论文可能是 router、负载损失、专家并行或规模实验；RAG 论文可能是 retriever、generator、索引或证据评估；Agent 论文可能是 prompting、搜索、工具协议、环境或 benchmark。

不同层级的主要指标不同。算子论文不能只用最终 benchmark，Agent 方法也不能只用单步准确率。先定位贡献层级，才能选择公平 baseline。

### 8.2 计算和数据预算是否公平

比较 Long Context 时，要固定模型、数据、位置编码和评估任务，说明长序列训练 token 是否增加。比较 MoE 时，要给出总参数、active FLOPs、训练 token、通信和存储。比较 RAG 时，要固定 generator、知识库版本、检索预算和上下文 token。比较 Agent 时，要固定模型、工具、最大步数、环境信息和候选搜索预算。

如果一个方法额外使用了更强的 reranker、更多候选、更多工具调用或更长输出，就要把这些资源写进账本。否则“方法提升”可能只是“预算提升”。

### 8.3 评估是否覆盖真实瓶颈

| 方向 | 不能只看 | 还要看 |
| --- | --- | --- |
| Long Context | 最大 token 和单点 needle | 位置、干扰、多证据、延迟和显存 |
| MoE | 总参数和最终 loss | active FLOPs、负载、drop、通信和稳定性 |
| RAG | 最终答案准确率 | recall、排序、证据支持、权限和时效 |
| Tool Use | JSON 合法率 | 工具选择、参数、执行、结果使用和副作用 |
| Agent | demo 成功一次 | 多次重复、步骤、恢复、成本和失败类型 |

### 8.4 Ablation 要能定位责任

系统论文应拆开验证模块：去掉 reranker、改 top-k、换 chunking、限制上下文、移除 memory、限制搜索深度、关闭工具重试、替换 router loss。每个消融都应保持其他预算和接口尽量一致。

若没有消融，读者无法知道提升来自核心机制、数据清洗、提示模板、额外计算还是评估泄漏。

## 9. Worked case：百万上下文的有效能力分析

### 9.1 表面结果

某模型接口接受 1M token 输入，长文档测试中可以找到位于开头和结尾的指定事实。团队据此宣传模型“具备 1M context understanding”。

这个结论过强，因为它没有报告训练长度、证据位置、干扰比例、跨段推理和请求成本。

### 9.2 分层测试

可以把测试拆为四层：

1. **接口层**：输入是否被 tokenizer 和 runtime 接受。
2. **定位层**：不同长度和位置的单事实是否找得到。
3. **组合层**：多个远距离事实是否能合并。
4. **决策层**：在冲突和无关信息中是否能依据证据给出正确结论。

假设定位层在开头/结尾成功率为 0.92，中间位置为 0.61，多证据组合为 0.48，且输入 token 成本是 128K 方案的 6 倍。更准确的结论是“接口支持 1M，定位能力随位置和任务复杂度下降，当前证据不足以证明通用 1M 推理”。

### 9.3 修复方案

修复可能包括长序列训练、位置分桶评估、证据重排、RAG 预筛选和上下文压缩。每种修复都要重新比较质量与成本，不能把“更长输入”本身作为目标。

## 10. Worked case：MoE 的质量提升来自哪里

一个 MoE 模型总参数是 dense baseline 的 4 倍，但每 token active FLOPs 接近 baseline。实验显示验证 loss 更低，然而 40% 的 token 被路由到两个热点专家，all-to-all 通信占 step time 的 28%，并有 3% token 因容量限制被丢弃。

此时不能直接写“MoE 用四倍参数换来更高效率”。需要至少做三组对照：

1. 固定 active FLOPs 比较 dense 和 MoE 的质量。
2. 固定总训练 token，改变 capacity factor 和 router loss。
3. 在相同拓扑下记录负载、drop、通信和端到端吞吐。

如果修复负载后质量保持提升、通信占比下降，才有证据说明专家容量对结果有贡献；如果质量提升消失，原始结果可能主要来自不公平的有效训练预算或数据路径。

## 11. Worked case：RAG 的答案正确但引用错误

某知识助手的最终准确率从 72% 提高到 81%，但人工审计发现引用支持率只有 55%。分析 trace 后发现，检索器召回了正确文档，generator 却用参数记忆补充了文档中没有的日期和条件；引用只放在段尾，读者误以为整段都有证据。

修复应拆开进行：对每个可核查 claim 标注证据 span，要求引用与 claim 对齐；对无证据内容要求模型标注不确定；对过期版本和权限文档加入过滤；再分别报告 answer accuracy、support ratio、citation precision 和拒答/澄清率。

这个案例说明“召回正确”和“回答有依据”是两个不同 claim。RAG 的证据链要比最终答案单分数更细。

## 12. Worked case：Agent 成功率与调用预算

某 Agent 在 100 个任务上完成 72 个，但平均每个任务调用工具 18 次，其中 12 个任务因重复重试造成超时。另一个简单 baseline 完成 66 个任务，平均调用 5 次，单位成功成本只有前者的 0.42 倍。

如果任务价值高且允许更长延迟，Agent 可能值得采用；如果是高并发客服，调用次数和 P95 延迟可能决定方案不可用。报告应同时给出成功率、步骤数分布、错误恢复率、超时、状态副作用和单位成功成本。

修复可以包括计划检查点、工具错误分类、幂等键、循环检测、预算上限和人工升级。减少调用不能成为唯一目标，否则可能过早停止而降低成功率。

## 13. 可运行的组合审计 demo

下面的纯 Python demo 使用合成记录，分别计算长上下文证据定位、MoE 负载、RAG 召回与引用支持，以及 Agent 工具执行成功。它不连接真实模型、索引或工具，只演示为什么一个总分需要拆成多个信号。

~~~python
from math import sqrt


def mean(values):
    return sum(values) / len(values) if values else 0.0


def population_std(values):
    center = mean(values)
    return sqrt(mean([(value - center) ** 2 for value in values]))


long_context = [
    {"position": "start", "success": 1},
    {"position": "middle", "success": 0},
    {"position": "end", "success": 1},
    {"position": "middle", "success": 1},
]
expert_load = [40, 18, 22, 20]
rag_queries = [
    {"relevant_retrieved": True, "claims": 2, "supported": 2},
    {"relevant_retrieved": True, "claims": 3, "supported": 2},
    {"relevant_retrieved": False, "claims": 1, "supported": 0},
]
agent_trace = [
    {"tool_ok": True, "final_ok": True},
    {"tool_ok": True, "final_ok": True},
    {"tool_ok": False, "final_ok": False},
    {"tool_ok": True, "final_ok": False},
]

long_recall = mean([row["success"] for row in long_context])
middle_recall = mean(
    [row["success"] for row in long_context if row["position"] == "middle"]
)
load_cv = population_std(expert_load) / mean(expert_load)
rag_recall = mean([row["relevant_retrieved"] for row in rag_queries])
support_ratio = sum(row["supported"] for row in rag_queries) / sum(
    row["claims"] for row in rag_queries
)
tool_success = mean([row["tool_ok"] for row in agent_trace])
agent_success = mean([row["final_ok"] for row in agent_trace])

signals = {
    "long_recall": round(long_recall, 4),
    "middle_position_recall": round(middle_recall, 4),
    "moe_load_cv": round(load_cv, 4),
    "rag_recall_at_k": round(rag_recall, 4),
    "citation_support_ratio": round(support_ratio, 4),
    "agent_tool_success": round(tool_success, 4),
    "agent_task_success": round(agent_success, 4),
}
actions = []
if signals["middle_position_recall"] < 0.75:
    actions.append("add_position_and_multi_evidence_tests")
if signals["moe_load_cv"] > 0.30:
    actions.append("repair_router_load_balance")
if signals["rag_recall_at_k"] < 0.80 or signals["citation_support_ratio"] < 0.80:
    actions.append("repair_retrieval_and_claim_citations")
if signals["agent_task_success"] < 0.80:
    actions.append("add_trace_recovery_and_budget_tests")
decision = "continue_after_system_repairs" if actions else "continue_to_shadow_eval"

for name, value in signals.items():
    print(f"{name}={value}")
print(f"actions={actions}")
print(f"decision={decision}")
~~~

在这组合成数据上，整体长上下文 recall 不能掩盖 middle-position recall，RAG 的召回也不能替代 citation support，工具调用成功更不能替代 Agent 最终任务成功。demo 的指标都不是模型能力的真实估计，而是设计评估分母的教学例子。

## 14. 常见失败模式与修复顺序

### 14.1 Long Context 最大长度很大但中间信息丢失

检查位置分布、训练长度、干扰比例、提示格式和是否只测 single needle。再做多证据、冲突证据和引用测试。若输入成本过高，比较 RAG 预筛选、上下文压缩和更长训练，而不是只增加 max position。

### 14.2 FlashAttention 让 kernel 变快但端到端没有提升

检查 prefill/decode 比例、padding、通信、KV cache、batch、kernel fallback 和数据搬运。算子加速被其他关键路径掩盖时，不能把局部 speedup 写成服务吞吐提升。

### 14.3 MoE 专家塌缩或通信成为瓶颈

检查每专家 token 分布、router entropy、capacity、dropped tokens、all-to-all 和拓扑。降低负载损失不一定解决热点；可能需要改变路由、容量、专家放置或 batch 组织。

### 14.4 RAG 召回好但答案不忠实

检查 context packing、证据位置、冲突文档、引用要求和生成模型是否使用参数记忆。增加 citation verifier、claim-level evaluation 和无证据时的澄清/拒答，而不是盲目增大 top-k。

### 14.5 Tool Use 参数合法但动作错误

把 schema validation、业务校验、权限检查和执行结果分开记录。对写操作增加 preview、审批、幂等键和回滚；对工具返回的外部文本做来源标记和注入隔离。

### 14.6 Agent 进入循环或错误级联

保存每一步 state、action、observation 和状态差分，区分瞬时错误、参数错误、权限错误和未知副作用。加入步数预算、重复状态检测、检查点和人工接管，不能只提高模型温度或让它“再想一遍”。

## 15. 如何复现一篇组合系统论文

### 15.1 先选择一个窄 claim

不要同时复现 Long Context、RAG 和 Agent 的全部功能。可以选择：

> 在相同 generator、证据集合和输入 token 预算下，reranker 是否提高多证据问题的 claim support ratio？

或：

> 在相同 active FLOPs 和训练 token 下，某种 router loss 是否降低 expert load CV 且不降低任务质量？

窄 claim 能让实验明确知道改变的是检索、路由还是生成。

### 15.2 固定接口和预算

至少固定：

1. 模型、tokenizer、位置编码和 checkpoint。
2. 数据、文档版本、索引构建时间和权限字段。
3. 输入/输出 token、top-k、候选数和最大 Agent 步数。
4. 工具 schema、版本、环境状态、错误注入和重置方式。
5. 计算精度、硬件、网络拓扑和 batch。
6. 指标定义、分母、失败状态和成本口径。

组合系统最容易因为隐藏预算不一致而产生虚假提升。一个 Agent 如果多允许十步，一个 RAG 如果多放三倍文档，一个 Long Context 如果使用更长输入，都应该在比较中显式记录。

### 15.3 证据分层

可以把复现结果分成四层：

1. **接口复现**：输入、工具或索引能够运行。
2. **机制复现**：观察到预期的 attention IO、router load 或 recall 变化。
3. **任务复现**：目标 benchmark 或真实任务指标改善。
4. **系统复现**：延迟、吞吐、成本、权限和故障恢复也符合主张。

只完成第一层不能声称完成论文主张。系统论文还要保存 trace、硬件和版本，否则别人无法解释差异来自算法还是环境。

## 16. 如何写这条论文线的研究报告

一份完整报告可以按以下叙事展开：

1. 问题是输入长度、模型容量、知识时效、外部执行还是多步可靠性。
2. 机制把能力从哪里引入，改变了哪条计算路径。
3. 需要什么额外数据、索引、专家、工具或环境。
4. 在什么预算下与哪个 baseline 比较。
5. 质量、效率、成本和安全指标的分母是什么。
6. 失败来自召回、路由、位置、协议、状态还是生成。
7. 哪些结果能外推，哪些只能留在当前模型和环境。
8. 下一步是优化模型、修数据、改系统、限制权限还是停止扩展。

例如，不能把“1M 输入通过”“RAG recall@10 为 0.9”“Agent 完成一个 demo”并列写成系统具备长程智能。应说明 1M 下的位置分布和任务成功，recall 是否覆盖所有所需证据，Agent 是否在受控环境中完成了真实状态变化，以及这些能力的单位成本。

## 17. 资料与证据边界

长上下文与位置机制可参考 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)、[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)、[FlashAttention](https://arxiv.org/abs/2205.14135) 和 [FlashAttention-2](https://arxiv.org/abs/2307.08691)。这些论文分别支持 RoPE、ALiBi、IO-aware exact attention 和并行实现的技术脉络；本章关于有效长度、位置分桶和生产成本的结论需要独立测量。

长上下文利用率可参考 [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) 和 [LongBench](https://arxiv.org/abs/2308.14508)。它们支持位置、任务和长文档评估的重要性；单一 needle 测试不能代表多证据推理或所有模型的生产行为。

KV cache 服务系统可参考 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。该论文支持分页式 KV cache 管理和 serving 吞吐的系统背景；具体收益依赖模型、硬件、batch、解码和 runtime 版本。

稀疏专家路线可参考 [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)、[GShard](https://arxiv.org/abs/2006.16668) 和 [Switch Transformers](https://arxiv.org/abs/2101.03961)。它们支持稀疏路由、专家并行、负载均衡和容量控制的论文脉络；本章的 active FLOPs 和 step-time 公式是教学记账，不是跨实现的精确成本模型。

RAG 的基础路线可参考 [REALM](https://arxiv.org/abs/2002.08909)、[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) 和 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)。它们支持可学习检索、稠密段落召回和参数/非参数记忆结合的研究背景；具体 chunk、索引、权限和引用结果仍需在目标语料上验证。

工具和交互推理可参考 [Toolformer](https://arxiv.org/abs/2302.04761)、[ReAct](https://arxiv.org/abs/2210.03629) 和 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)。这些论文支持工具调用、reasoning/action 交替和搜索式 thought 的研究方向；它们不自动证明生产工具调用安全，也不能把受控 benchmark 的成功率外推到任意 Agent。

Agent 评估可参考 [WebArena](https://arxiv.org/abs/2307.13854) 和 [AgentBench](https://arxiv.org/abs/2308.03688)。它们支持在环境和多任务上测量 Agent 的必要性；真实业务还要加入权限、状态副作用、隐私、成本、回滚和人工接管。

本章的 worked case、Python demo、合成负载、检索结果、Agent trace 和决策字符串均为教学构造。真实复现实验应保存模型配置、训练长度、位置方案、文档 snapshot、索引版本、router 统计、工具 schema、环境状态、完整 trace、硬件信息和失败记录，并明确区分论文事实、项目实测和教学近似。

## 18. 结语：从单次生成到可复查的系统能力

Long Context 让模型看到更多输入，但有效能力取决于位置、干扰、训练、推理成本和多证据利用。MoE 让模型拥有更大总容量，但 active compute 之外还有路由、负载、通信和存储。RAG 让知识可更新、可引用，但检索和证据链成为新的责任边界。Tool Use 让模型连接计算和业务系统，但结构化参数不等于授权动作。Agent 把任务展开为状态循环，但长程可靠性取决于恢复、停止、权限和成本。

阅读这条论文线时，不要只问“哪个方法更强”。更好的问题是：能力从哪里来，预算转移到了哪里，哪个模块对失败负责，论文的 baseline 是否公平，评估是否覆盖真实瓶颈，以及在什么边界内结论成立。

当研究者能够把模型、上下文、专家、检索、工具和环境拆成不同对象，再用任务成功、证据支持、路由负载、调用轨迹、延迟、成本和安全状态共同验证时，这些方向才不再是新闻式名词，而成为可以复查和迭代的系统工程。
