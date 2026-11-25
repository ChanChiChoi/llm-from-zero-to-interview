# 第 81 章 DeepSeek V4.1-Flash：CED、CSA2、缓存重算与原生多模态

本章以 2026-10 更新的 DeepSeek-V4.1-Flash-1017 为当前模型身份，以 2026-09 技术报告和两个固定配置快照解释可公开核验的架构结构。中心问题是：长上下文输入、逐 token 输出、global 历史、局部窗口状态和图像表示怎样分工，以及 CED、CSA2、近似重算和低精度分别改变什么计算或存储账本。读者应能从模型层配置追踪到状态如何产生、共享、检索、恢复或写入；不能把报告估算、发布方质量成绩、checkpoint 字段、推理框架实现和生产部署效果混为同一证据。第 77 章承接 V4 的 CSA/HCA 与异构 cache，第 78 章完整讲 mHC，第 80 章讲 MTP 与 speculative decoding；本章解释 V4.1 的变化并回链前置，不重复通用机制、API 协议或完整 serving engine 操作。

## 81.1 模型身份与架构总图

本节确定模型版本、架构对象和本章的比较口径。V4.1-Flash-1017 是 2026-10-17 更新的 checkpoint；其官方公开配置在本章使用的结构字段上与 2026-09 初始发布配置相符，而技术报告和原始评测主要对应初始发布版本。读者应将 backbone 参数、Engram 参数、prefill/decode 激活参数、KV 字节与上下文长度视为不同量，并在后文按来源分别核对；不能把自动参数统计、服务 API 名称或发布宣传替代固定模型身份。

### 81.1.1 0909 技术报告与 1017 checkpoint 的证据对象

说明 2026-09 技术报告的层布局、压缩比与实验属于初始 V4.1-Flash；官方 changelog 将 1017 说明为后续模型更新并报告新成绩。用初始固定配置与 1017 固定 release 配置对照本章涉及的结构字段，指出字段相符支持架构配置延续，但不证明后续训练数据、权重或运行时完全相同。报告、固定配置、API 指向和部署测试各自回答不同问题。

### 81.1.2 参数量、激活量、cache 与长度不能互相换算

模型卡/技术报告将 552B 标作 backbone 参数、196B 标作 Engram 参数，并将 8B/16B 分别报告为 prefill/decode 每 token 激活参数；global KV 的 890 bytes/token 和 1M 上下文则是另外两类口径。Hugging Face 文件页自动显示的参数总数与正文分类口径未获来源解释，应并列保留，不自行算成内存、FLOPs 或速度结论。

## 81.2 CED：输入与输出为何走不同计算路径

本节把 Causal Encoder-Decoder（CED）放回长输入、较短输出的工作负载中，解释它如何把 prefill 时生成 global KV 的来源与 decode 主干分开。与标准 decoder-only 的差别不是简单把双向 encoder 接到 decoder 前面：前半仍使用 causal Transformer，decoder 的 global KV 从最后 encoder hidden state 投影；层级 SWA 状态仍需逐层产生。读者应能区分 global 与 local 路径，并理解“约减半 prefill”依赖长序列假设及局部状态处理。

### 81.2.1 Decoder-only 基线如何建立 prefill 与 decode 状态

说明输入 prefill 处理已有 token 并生成后续 decode 要读取的历史状态；生成阶段每个新 token 继续经过主干并访问历史 KV。区分输入 token 数、输出 token 数以及 cache 命中/缺失，使读者看出为何 agent 类长输入任务会对 prefill 与 cache 成本敏感。

### 81.2.2 Decoder 的 global KV 来自最后 encoder 层，SWA 仍逐层计算

解释 40 层被分成 20 层 causal encoder 与 20 层 decoder；decoder 各层使用不同投影，将第 20 层 hidden state 映射成各自 global KV 与压缩权重。另说明所有层的 local SWA K/V 仍从本层 hidden state 生成，因此 CED 不是把后 20 层完全删掉，也不是把所有 cache 变成共享的一份。

### 81.2.3 Prefill 计算近似及其短轮次边界

用报告的 $O(NL/2+n_{\mathrm{win}}L/2)$ 近似表达长输入时的 encoder 计算与 decoder SWA replay 项，并说明 $N\gg n_{\mathrm{win}}$ 时才可将其读作约一半的 prefill 计算。该式是模型报告中的计算近似，不是 latency 定律；短上下文、多轮短后缀和实际 kernel 路径要单独测量。

## 81.3 CSA2：跨层共享 global KV 与稀疏选择

本节围绕 attention 层在深度方向共享什么状态展开。Compressed Sparse Attention 2（CSA2）仍让每层生成自己的 global query 与 SWA KV，但依不同静态模式生成、复用 main KV、indexer K 和 Top-K 位置。它将时间/序列压缩与跨层缓存复用结合起来；应与第 77 章 V4 的 CSA/HCA hybrid 及第 74 章压缩读取机制区分。

### 81.3.1 一个 query 使用哪些历史表示

定义 main KV、indexer K、indexer query、Top-K indices 与 layer-local SWA KV 的职责关系。说明 CSA2 的压缩率 $m$ 控制主 KV 的序列条目密度，而 Top-K 控制 query 读取哪些压缩条目；索引打分与注意力读取是相关但不同的动作。

### 81.3.2 Full、Reindex、Reuse 三种模式改变哪些状态

对齐三种模式的输入、计算与输出：Full 产生 main KV、indexer K 与新 Top-K；Reindex 复用 main KV/indexer K、用本层 indexer query 刷新 Top-K；Reuse 复用当前共享 main KV 和最新 Top-K，不运行新索引。三者均保留本层 global Q 和 SWA K/V，并在所选 global 条目与本地窗口上计算 attention。

### 81.3.3 1017 公开配置中的 encoder/decoder 层排布

按报告和固定配置读取：encoder 前两层为 SWA-only，其后 18 层以 $m=2$ 分为三组 `Full + 5 Reuse`；decoder 的 20 层为 $m=1$，第一组 `Full + 3 Reuse`，之后四组 `Reindex + 3 Reuse`。说明这是该 checkpoint 的排布，不是 CSA2 的固定通用模板；压缩 ratio 字段本身不能替代模式来源核验。

## 81.4 Hierarchical Sparse Indexer：先建候选池，再让深层重排

本节解释 decoder 内的层次化稀疏索引怎样控制反复扫描长历史的成本。第一个 Full layer 对当前 query 的因果可见范围执行全局评分，并由块内最大分数选出候选 blocks；后续 Reindex layers 只在候选位置内重新评分，Reuse layers 沿用最近 Top-K。候选池限制只约束后续索引范围，不消除初始全范围扫描、KV 维护、SWA 或其他长上下文成本。

### 81.4.1 2,048 个候选块如何形成 16,384 个候选位置

说明报告中的 blockwise candidate selection：每 8 个位置形成一个块，最多选 2,048 个块，因此后续索引池最多含 16,384 个位置；最终 query 仍从池内选不超过 Top-512。将候选 pool 与最终读取集合分别说明，避免读者把两个 top-k 误作同一层筛选。

### 81.4.2 固定池缩短后续评分，但引入不可恢复的漏检路径

说明固定池时 Reindex 的评分量不随完整上下文长度线性增长，但 Full layer 仍扫完整可见范围。若相关证据未进入候选池，后续重排不能找回；若只分析最终 Top-K recall，也会遗漏第一层候选选择的损失。报告没有给出通用 recall 保证，教学乘积只可在明确的简化假设下用于说明误差传播。

## 81.5 SWA Bounded Replay：用有界重算换长期存储

本节区分滑动窗口状态的短期作用与 global KV 的长期前缀复用。V4.1 报告的 SWA Bounded Replay 在局部状态丢失时只重放最近 $n_{\mathrm{win}}$ 个 token，以降低持久化 SWA cache 的占用；由于跨层 SWA 依赖延伸到重放窗口之外，这是一种近似恢复，不是无损 cache restore。内容限定于报告所述的 encoder/decoder 恢复路径和存储—计算权衡，不扩展成通用 cache scheduler 教程。

### 81.5.1 精确恢复为何会跨层累积窗口依赖

说明某层窗口内表示依赖更低层此前生成的 hidden state；逐层精确重建 $L$ 层的窗口状态需要比单层 $n_{\mathrm{win}}$ 更多的历史计算，报告将其近似记作 $L\times n_{\mathrm{win}}$ tokens。对照 Bounded Replay 截断窗口的做法，明确结果状态与完整前向并非数学上完全一致。

### 81.5.2 Encoder replay 复用 global KV，只重建局部 SWA 状态

说明在前缀 global KV 命中而 SWA 状态缺失时，系统重放前缀最近窗口并与未缓存后缀衔接；重放段复用已缓存 global KV，不重写 global cache，未缓存后缀继续产生所需 global 与 local 状态。作者报告后续响应质量影响很小，结论仅支持所报告的模型与设置。

### 81.5.3 Decoder replay 让 prefill 尾部生成 decode 所需的 SWA 状态

说明 decoder global KV 已可由 encoder hidden states 投影，但 decode 初段仍需要 decoder 各层的本地 SWA KV；报告在每次 prefill 时回放 prompt 尾部，为后续生成重建该局部状态，且不将这段近似 decoder 状态当作长期前缀 cache。对多轮短后缀而言，这一固定 replay 开销仍需单独核算。

## 81.6 FP4 global KV：存储精度与字节账本

本节把 main KV 的低精度存储与 attention 计算分开。报告使用 E2M1 FP4 值及每 16 个 channel 一个 E4M3 scale，量化在 RoPE 后存入 cache、attention 前再反量化；SWA KV 保持 FP8。FP4 的首要作用是降低 cache 字节，而不是保证所有硬件上的矩阵运算更快；量化误差必须与检索/任务质量一起评估。

### 81.6.1 分组 scale 怎样构成 FP4 cache 值

用 $\hat{x}_i=s_{g(i)}q_i$ 表达量化值由低位 payload 和分组 scale 共同还原，解释组大小对 scale 开销和误差范围的影响。指出该模型的 QAT 时序以报告详细小节为准；公开材料对早期概述和详细章节的阶段措辞略有差异，不将其扩写成从头预训练均使用 FP4。

### 81.6.2 890 bytes/token 的分母、组成与不包含项

将 890 bytes/token 标注为 V4.1-Flash 的 global KV footprint，并按报告口径与 V4-Flash 约 4 倍缩减对照；将 persistent KV 约 1/8 的比较与 SWA 不再进长期持久 cache、global KV 进一步缩减联系起来。明确这些数不是全部权重、临时 SWA、indexer、batch、通信和 workspace 的机器显存总和。

### 81.6.3 低精度 cache 的质量与系统验证指标

列出应联合观察的质量、候选召回、长距离证据检索、dequant 时间、cache bytes、带宽和延迟；区分论文中的格式选择与具体硬件 kernel 的实测。没有同一模型/负载/precision 的对照时，不把字节减少自动换算为性能收益。

## 81.7 MoE 与 Engram：计算专家和条件记忆的不同作用

本节只解释理解 V4.1 结构必需的参数与记忆路径，MoE 的通用路由机制回链第 21 章。MoE 用少量 routed experts 执行 token 相关 FFN，Engram 则通过 token-based lookup 访问模型内条件记忆；二者都不同于当前请求 KV，也不同于可独立更新的外部检索库。官方报告和模型卡将 552B backbone 与 196B Engram 分开报告，页面自动参数统计尚无可核实的统一对账方式。

### 81.7.1 384 routed、1 shared、top-6 的含义和边界

说明每个 MoE 层的专家数和 token 路由数分别描述可选容量与参与计算的 routed 分支；总权重存储、active parameters、all-to-all 通信、负载不均及 cache 是不同账本。模型的 8B prefill/16B decode 激活口径不能从 top-6 单独推出。

### 81.7.2 Engram 用 n-gram 哈希访问模型内条件记忆

说明 Engram 将 token 序列的 2/3/4-gram 地址映射到 embedding 表，并在两处层位置读取条件信息；报告公开的多头 hash、embedding 维度与 FP8 设置可用于说明表结构。区分它与序列 KV、外部 RAG 的地址来源、生命周期和更新方式；未公开的 hashing/runtime 细节保留为未知。

## 81.8 Single-Pass mHC：用前一层系数消除融合依赖

本节只解释 V4.1 对多路残差流部署计算的改动，并将 mHC 本身的双随机矩阵性质留在第 78 章。原路径需要先由当前残差状态计算输入混合系数，再读取残差完成输入混合，形成跨 kernel 依赖；Single-Pass 使用前一 block 预测的输入混合系数，使残差读、系数预测和混合输入可合并处理。报告将这一变化与部署 Mega-mHC kernel 的激活流量减少相联系，仍需区分实现带宽收益与端到端质量/速度。

### 81.8.1 原 mHC 的系数依赖怎样阻止单遍融合

引用第 78 章对多路残差状态与系数 $A_l,B_l,C_l$ 的定义，说明当 block $l$ 使用当前状态计算的 $A_l$ 读入同一状态时，为得到完整归约会形成额外遍历依赖。

### 81.8.2 将 $A_l$ 延迟到下一层如何降低激活流量

说明 Single-Pass 让当前 block 使用已存在的 $A_{l-1}$，其余残差更新与系数预测得到可融合的数据路径。报告称部署融合流量由原实现约 $(4n+4)d$ 降到 $(2n+2)d$；该数绑定作者实现与流数，不能当作所有硬件的端到端加速比。

## 81.9 DSpark：半自回归草稿仍需目标模型验证

本节解释 V4.1 的 DSpark 草稿与推测解码，而不将“同时预测多个位置”误作“多个输出已被接受”。DSpark 用三层 draft network、Markov 依赖头和 confidence head 产生并调度若干候选，再由目标模型验证可提交前缀；验证长度受草稿质量、接受情况和当前引擎吞吐影响。第 80 章承担 MTP 与推测解码的一般执行语义，本节聚焦 V4.1 专属结构和训练边界。

### 81.9.1 三层 draft、五个位置与 confidence-scheduled verify

说明一次 draft forward 并行提出五个位置，Markov head 建模位置间依赖，confidence head 预测条件接受概率；调度器结合 engine throughput 曲线选择验证长度。输出正确性仍由 target verification 保证，草稿数量本身不是实际接受长度或加速因子。

### 81.9.2 DSpark 的训练时序与 MTP-3 的区别

报告称 DSpark 在 backbone pretraining 后单独训练，post-training 阶段可继续与 backbone 一同训练但 DSpark objective 不向 backbone 反传；该模型 backbone pretraining 不使用 MTP。对比第 80 章 Step 3.5 的多 token prediction 训练目标，说明相似的预测位置数量不表示同一训练机制。配置字段和类入口只能证明 artifact/runtime path 存在，不能证明接受率与端到端 SLO。

## 81.10 原生视觉输入：图像特征怎样进入因果语言主干

本节围绕图像从像素 patch 到语言模型序列表示的结构路径展开。DeepSeek-ViT 产生空间视觉特征，经 3×3 pixel-unshuffle 降采样和 MLP projector 后，对齐到图像 token 位置，与文本 embedding 一起进入 causal backbone；原生多模态表示联合建模路径，不代表 API 图片预算、文件权限、工具调用或外部媒体协议。

### 81.10.1 DeepSeek-ViT、2D-RoPE 与 3×3 pixel-unshuffle

说明视觉编码器基于 ViT、为可变分辨率使用 2D-RoPE，并以 RMSNorm/SwiGLU 和线性 patch embedding 对齐模型设计；pixel-unshuffle 将相邻 3×3 空间位置重排进通道，使视觉 token 数约缩为原来的 1/9。模型配置给出的层数、patch size 和 token 上限是该 revision 字段，不替代任意图片输入下的真实 token 预算。

### 81.10.2 Projector 与文本共享序列后的模态路由

说明 projector 将视觉特征映射到语言主干 hidden dimension，插入图像 token 对应位置后共同经过语言层；报告为文本和图像 token 维护独立专家负载 correction bias，防止合并统计掩盖单一模态负载不均。区分架构输入路径与宿主执行权限。

## 81.11 效率与质量证据：不能把整机结果归给单个组件

本节将结构、模型质量与系统效率证据按可支持的主张分开。技术报告将 45T 多模态预训练、64K 稀疏注意力训练以及 34T token 后扩展至 1M 与模型成绩放在同一方案中；报告还估算长上下文 decode FLOPs 和 KV 存储。由于 CED、CSA2、精度、数据、Engram、mHC、DSpark 与训练管线共同变化，没有单项结果证明某个机制独自带来总能力或服务性能提升。

### 81.11.1 KV、decode FLOPs、延迟和质量属于不同度量

整理 global KV 890 bytes/token、persistent KV 相对 V4-Flash 约 1/8、4K 到 1M context 时报告的 decode FLOPs 增量约 1/4 等数据；标出各自的口径和模型比较条件。强调这些不是 TTFT、TPOT、wall-clock 吞吐、单位任务成本或独立复现结果。

### 81.11.2 1017 更新成绩不证明架构收益的来源

选取官方 1017 与原始模型卡/官方更新页的少量任务结果，说明更新模型声称提高 agent/视觉成绩并减少任务时间或输出长度；成绩仍依赖 benchmark、harness 与 effort。由于 1017 不是对 CED、CSA2 或 FP4 的单因素消融，不能从新旧分数差估算某个结构的增益。

## 81.12 面试追问：从状态变化走到成本和证据

本节把章节机制转化为可讨论的问题，要求回答者能指明每种结构改动的输入、状态、动作和结果，并识别其适用条件与证据边界，而不是复述模型宣传数字。

### 81.12.1 CED 与 CSA2 各自减少哪部分工作

围绕 prefill global KV 的复用、逐层 SWA 生成、跨层 main KV 共享、索引刷新与 query 读取，讨论哪些计算/存储项减少、哪些仍保留，以及怎样选择可公平比较的基线。

### 81.12.2 Bounded Replay 用什么近似换取什么资源

追问为什么有界重放不等于精确恢复，如何比较 cache persist、host memory、重算 token、恢复质量与尾延迟，以及哪些 manifest 状态需要固定。

### 81.12.3 1017 的指标变化能支持哪些结论

要求区分配置字段、技术报告自报、官方 benchmark update、第三方 runtime 和目标硬件实测；说明为何模型整体变化不能归因于单个注意力或 cache 组件。

## 81.13 练习：复算结构账本并设计验证

练习围绕公开固定配置和作者报告，帮助读者检查层分组、候选池、KV 口径与 replay 成本；不要求本地下载或运行完整 552B 权重，也不将合成练习结果写成模型性能。

### 81.13.1 从层表复原 CSA2 分组

根据 40 个 backbone 层与模式来源，逐组标出 encoder/decoder 中 Full、Reindex、Reuse 的 layer id，区分 ratio=0 的 SWA-only 层与 DSpark draft 配置项。

### 81.13.2 拆分两级 Top-K 的候选预算

根据 2,048 个 8-position blocks 与最终 Top-512，分别计算候选池最大位置数和最终读取数；再构造候选池漏检的示例，说明仅测最终 Top-K recall 会遗漏什么。

### 81.13.3 比较 SWA 持久化与有界重放方案

对相同请求长度和窗口长度建立符号成本表，分别记录持久化字节、读取、replay token、cache miss 和恢复检验。设计至少一个会使 replay 状态不一致的版本、位置或边界变化，并写出应观察的指标。

## 81.14 来源与前置章节

正文来源以 DeepSeek-V4.1-Flash 技术报告 v1、原始 checkpoint 固定配置、1017 官方模型卡与固定配置、官方 1017 changelog 为主；TRAIL Labs 技术解读和 Stanford KV/cache notes 只用于教学组织参考。回链第 74 章 CSA/HCA、第 77 章 V4 配置、第 78 章 mHC、第 80 章 MTP/speculative decoding，以及第 21 章 MoE；推理引擎完整操作留给第 6、24 册。
