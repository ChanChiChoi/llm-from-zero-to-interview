# 前沿架构与模型更新：机制轴与披露快照

> 本章更新于 2026-10-30，提供机制分类、评测入口和有日期的代表性披露快照，用于快速定位相关专题；覆盖范围不承诺穷举全部新模型。模型机制深入第二十一册，产品接口事实以发布方文档为准。

## 先判断变化落在哪条轴上

讨论新架构时，先把“序列位置”和“网络深度”分开。序列位置表示输入里的第几个 token；网络深度表示第几层。标准自注意力（self-attention）主要决定当前 token 从哪些历史位置读取信息；键/值（key/value）缓存，常写作 KV cache，会保存每层供后续查询的历史状态。残差连接则把前面层的表示传到后面层。因此，减少历史状态、改变历史层的聚合方式、增加残差分支和增加专家容量，分别处理不同瓶颈。某个模型可以同时采用其中几种机制，但它们的收益不能只凭一个总分或“长上下文”标签互相代替。

序列方向的一端是显式注意力：较完整地保留每个位置的键和值，查询时再按内容寻址；代价会随上下文长度增加。另一端是固定形状的递归状态：每个新 token 更新一份紧凑状态，运行账本较小，但状态更新必须把旧信息压进有限容量。Kimi Delta Attention（KDA，Kimi 增量注意力）使用通道级衰减与 delta 写入；Kimi Linear 把 KDA 路径与多头潜在注意力（Multi-head Latent Attention, MLA）组合，分别承担压缩记忆与显式寻址。由“状态固定”只能推出状态张量形状不随前缀增长，不能推出精确长程检索一定保持。

压缩注意力把完整历史先变成较少的条目，再由索引器选择候选。这个设计要分开检查两个环节：压缩有没有保住问题需要的细节，选择器有没有找回含证据的条目。DeepSeek V4 的压缩稀疏注意力（Compressed Sparse Attention, CSA）在压缩条目上做稀疏选择；高度压缩注意力（Heavily Compressed Attention, HCA）使用更强压缩并对压缩结果做全量注意力（dense attention）。Qwen3.8 的 Qwen Sparse Attention（QSA）也压缩微块（micro-block）并稀疏选择，但其位置编码、候选预算、训练方式和缓存状态是模型专属设计。它们同属“先压缩或筛选历史”的问题域，不能因名字相似就把实现和复杂度视为相同。

深度方向关注层与层之间怎样复用表示。DenseFormer、深度交叉注意力（DeepCrossAttention, DCA）和注意力残差（Attention Residuals, AttnRes）都改变历史层输出的聚合方式：DCA 用输入相关的深度交叉注意力重组层表示，AttnRes 用 softmax 权重选择先前层输出，并用 Block AttnRes 组织块间选择。它们沿网络深度路由信息，不能直接解释 token 维 KV cache 的下降。残差流扩展则让多路状态同时经过网络；流形约束超连接（Manifold-Constrained Hyper-Connections, mHC）把流间线性混合映射约束到双随机矩阵集合，门控残差（Gated Residual, GR）使用分支读写门。二者处理多路残差的混合或门控，约束范围都不覆盖整个非线性网络的激活和梯度。混合值注意力（Mixture-of-Value Attention, MoVA）将专家路由放到注意力的 value 路径，N-gram Embedding 为短程 token 组合增加查表容量；它们改变的是表示容量与计算分配，不是远程证据检索。

下表把每条轴对应的观察问题和完整讲解位置放在一起。表内机制描述只用于定位；模型级实现、公式和实验条件应以对应专题的来源为准。

| 观察的瓶颈 | 机制轴与代表方法 | 首要失效线索 | 深入阅读 |
| --- | --- | --- | --- |
| 深层表示被固定残差累加，重要层输出不易单独调用 | 深度方向聚合：AttnRes、DCA | 层输出/梯度随深度失衡；增加上下文无法修复 | [第二十一册第 75 章](../../book-21-transformer-architecture-evolution/chapters/75-attention-residuals深度方向注意力.md)；[AttnRes 原文](https://arxiv.org/abs/2603.15031)；[DCA 原文](https://proceedings.mlr.press/v267/heddes25a.html) |
| 长前缀的显式 KV 随 token 增长 | 递归状态：KDA；混合显式与递归记忆：Kimi Linear | 精确名称召回、远距离键值关联或状态恢复失败 | [第二十一册第 76 章](../../book-21-transformer-architecture-evolution/chapters/76-kda从delta规则到线性注意力.md)；[Kimi Linear v2](https://arxiv.org/abs/2510.26692v2) |
| 历史 KV 成本过高，需减少候选数量或表示大小 | 压缩与索引：CSA/HCA、QSA | 压缩重建误差、候选召回、尾部恢复需分别观测 | [DeepSeek V4 技术报告](https://arxiv.org/html/2606.19348)；[第二十一册第 77 章](../../book-21-transformer-architecture-evolution/chapters/77-csa-hca从压缩kv到百万上下文.md)、[第 83 章](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md) |
| 多条残差流的映射偏用、放大或衰减某一路 | 流间混合约束：mHC；分支门控：Gated Residual | 流占用、混合矩阵、梯度/激活与训练开销 | [第二十一册第 78 章](../../book-21-transformer-architecture-evolution/chapters/78-mhc双随机残差连接.md)、[第 83 章](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)；[mHC v2](https://arxiv.org/html/2512.24880v2) |
| 需要为 token 的 value 表示或短程重复模式增加容量 | Value 路径专家：MoVA；局部查表：N-gram Embedding | 路由负载/通信，或随机查表带宽与容量 | [第二十一册第 82 章](../../book-21-transformer-architecture-evolution/chapters/82-k2-horizon-mova-36b-a4b与uno.md)、[第 83 章](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md) |

2026 年的 Multi-Stream Attention Residuals（MSAR，多流注意力残差）预印本摘要把多头深度路由作为 AttnRes 的扩展方向。现有摘要证据足以描述这一研究方向，但不足以比较通用效果或训练开销；[MSAR 预印本](https://arxiv.org/abs/2610.13630)的具体实验数字应结合完整论文与实验条件解读。

## 用一个小例子区分压缩损失和候选漏选

假设教学输入有 16 个历史 token，每连续 4 个 token 压成一条记录，于是得到 4 个候选块。若一个稀疏索引器只保留得分最高的 2 个候选（top-2），而关键定义所在的第 3 块排名第三，读取阶段就没有该块，精确答案会漏掉。把预算从 2 增至 3，在这个构造例中可以把第 3 块纳入候选；这只解决选择器漏选。

现在看另一种情况：压缩第 3 块时已经丢掉定义里的限定词。即使后续对全部 4 个压缩条目做 dense attention，模型仍只能读取压缩后的表示，不能从中还原未保留的原 token。因而，扩大 top-k 改善的是候选召回；提高压缩保真度、保留近期窗口或正确恢复未完成尾部，分别处理压缩与缓存生命周期问题。这里的 16、4、2 和“排名第三”是手算用的构造值，不是任何模型的配置或实测。

这组数字也提示如何拆分实验。先用含精确名称和定义的问题测候选召回；再固定候选集合，比较压缩前后的答案所需字段是否仍可恢复；最后测窗口、尾部、重启恢复和实际 cache bytes。若只看最终准确率，选择器、压缩器、位置机制和缓存状态的错误会混在一起。若只看上下文长度，则无法知道模型是否找到了相关证据。

### 把 KV cache 的量级算出来

对于保存完整 key/value 的教学模型，设有 $L$ 层、$T$ 个历史 token、每层 $H_{kv}$ 个 key/value head、每个 head 的维数为 $d_h$，每个标量占 $b$ bytes。忽略对齐和元数据时，原始缓存大小近似为 $M=2LTH_{kv}d_hb$；因子 2 分别计 key 和 value，变量均为正整数，$b$ 表示实际存储精度的字节数。

取构造值 $L=2,T=16,H_{kv}=2,d_h=4,b=2$（每个标量 2 bytes）：单个 token 在一层的 key 占 $2×4×2=16$ bytes，key 加 value 为 32 bytes；16 个 token 为 $16×32=512$ bytes/层；两层合计 $2×512=1,024$ bytes。这个数只计算一个请求的原始张量，不含 block 对齐、量化 scale、索引、workspace、批处理和 allocator 碎片，不能当作真实服务显存。它显示显式 KV 的账目随 $L$ 与 $T$ 成比例变化；固定大小的递归状态改用状态维数和精度计账，而压缩注意力还要把索引、尾部和未压缩窗口分别加入。深度残差聚合关心的是同一 token 在不同层之间的表示传递；即使序列长度保持 16，增加或保存更多深度状态也可能改变内存和通信，所以不能把“层数开销”和“上下文长度开销”混成一个数字。

## 按失败现象设计诊断

| 观察到的失败 | 先隔离的机制 | 可复核的测量 |
| --- | --- | --- |
| 文中出现过的专名或定义仍无法精确回答 | 显式 KV、递归状态、压缩条目、索引候选与位置编码 | 同一位置和长度的 exact-match/recall；候选召回率；压缩前后字段保留；cache bytes |
| 增加上下文后仍漏掉中间信息 | 长距离寻址和 lost-in-the-middle，而非仅看窗口上限 | 固定证据内容，改变证据位置、长度、干扰项和查询形式，逐切片报告召回 |
| 深层训练不稳定或层输出长期失衡 | 深度聚合、残差流混合、归一化与优化器 | 固定数据、token 数、精度、初始化和算力后比较层输出/梯度统计、loss 曲线与通信 |
| 专家模型速度不随每 token 激活参数变化 | 路由、dispatch、负载倾斜、权重驻留与后端 kernel | 同一硬件和后端报告吞吐、每输出 token 时间（TPOT）、峰值显存、通信、专家负载分布及失败率 |

诊断要使用能区分候选解释的输入。比如同一条长文档分别把证据放在开头、中间和末尾，固定查询和干扰段，再逐位置记录召回；若只换一份文档，模型差异、证据位置和内容难度会一起变化。缓存恢复则可固定相同 token 前缀，分别测冷启动、命中、驱逐后恢复和尾部未封块，逐步比较状态摘要与最终答案。混合专家（MoE）服务要在相同请求分布下看专家负载的中位数和长尾；平均每 token 激活参数一样，少数热点专家的排队仍可能拉高 p95（95 分位）延迟。把输入、状态、请求和硬件条件固定，诊断表才会指向可复现的机制差异。

比较结构时，必须把参数总量与每 token active 口径分开，把模型权重和 KV/递归状态分开，把论文实验与线上接口测量分开。对论文方法，记录模型规模、训练 token、基线、数据、精度、并行拓扑和指标；对 API 产品，记录 model ID、平台、请求配置、日期、限流和工具状态。只有这些条件足以对齐时才作数值横比；否则写清各自回答的问题。固定状态大小、百万级上下文、较低 active 参数或单一 benchmark 分数，都不能单独推出目标部署的检索质量、速度、并发或成本。

## 代表性模型披露快照（截至 2026-10-30）

本表刻意区分架构资料、接口公告和系统卡摘要。链接应按页面日期和具体版本理解；未披露字段保持未知。

| 模型 | 官方来源支持的事实 | 能得出的结论与边界 |
| --- | --- | --- |
| Gemini 4 Argon | Google DeepMind 模型卡将其描述为稀疏混合专家（Mixture of Experts, MoE）Transformer，并列出原生文本、视觉和音频输入；未给出参数量或完整层配置。[模型卡](https://deepmind.google/models/model-cards/gemini-4-argon/) | 可记录为模型卡公开的结构类别与输入模态；路由细节、参数和推理实现仍未知。详解见[第二十一册第 95 章](../../book-21-transformer-architecture-evolution/chapters/95-gemini-4-argon长程推理与网络安全防御.md)。 |
| Mistral Large 4 | Mistral 模型文档列出 1.05T 总参数与 52B 每 token 激活参数（active parameters）；NVIDIA 官方模型页把 49B 说明为每 token 专家参数，并将约 52B 计为包含 embedding 和输出层的口径。[Mistral 文档](https://docs.mistral.ai/models/mistral-large-4)、[NVIDIA 模型页](https://docs.api.nvidia.com/nim/reference/mistralai-mistral-large-4-1t-a52b) | 数字必须连同版本和统计口径记录；总参数、专家每 token 计算量与含 embedding/输出层的 active 口径不是同一个字段。来源未支持把 active 参数直接换算为 serving 速度。 |
| GPT-6.1 Terra | OpenAI 发布说明聚焦视觉推理与 computer use；API 模型页列出 1,050,000 context 和 128K 最大输出。[发布说明](https://openai.com/index/introducing-gpt-6-1-terra/)、[API 模型页](https://developers.openai.com/api/docs/models/gpt-6.1-terra) | 这些是发布定位和接口容量字段。上述页面没有披露内部层结构、参数量或训练配方，不能用接口能力填补架构信息。 |
| Claude Haiku 5.5 | Anthropic 的 2026 年 10 月官方系统卡摘要列出文本和图像输入，并概述预训练及后训练；官方产品页提供发布信息。[Transparency Hub](https://www.anthropic.com/transparency)、[产品页](https://www.anthropic.com/claude-haiku-5-5) | 摘要没有给出精确架构或参数规模；应把它作为有来源但信息粒度有限的系统卡披露，不推断未公开的网络结构。 |

这几项仅示范怎样按来源类型读模型发布。Gemini 4 Argon 的深入架构说明在本册上方链接；其他第二十一册模型专题也覆盖 Mistral Small 4、Step 3.5 Flash、DeepSeek V4/V4.1、K2 Horizon、Qwen3.8、GLM-5.x、GPT-OSS、Kimi K3/K2.x 和 Qwen3 系列。产品价格、alias、上下文上限与模型生命周期会变化，应直接检查发布方当前文档；模型卡没有披露的层数、路由或训练细节应保留为未知。

## 第二十一册架构专题入口

下表用稳定章节承接模型家族、具体机制和系统代价。每章的模型规格与实验数字都有各自版本和证据范围，不能合并成同一榜单。

| 主题 | 深入阅读 |
| --- | --- |
| AttnRes 与深度方向选择 | [第 75 章](../../book-21-transformer-architecture-evolution/chapters/75-attention-residuals深度方向注意力.md) |
| KDA 与线性注意力 | [第 76 章](../../book-21-transformer-architecture-evolution/chapters/76-kda从delta规则到线性注意力.md) |
| CSA/HCA 与压缩 KV | [第 77 章](../../book-21-transformer-architecture-evolution/chapters/77-csa-hca从压缩kv到百万上下文.md) |
| mHC 双随机残差连接 | [第 78 章](../../book-21-transformer-architecture-evolution/chapters/78-mhc双随机残差连接.md) |
| Mistral Small 4；Step 3.5 Flash | [第 79 章](../../book-21-transformer-architecture-evolution/chapters/79-mistral-small-4混合推理与eagle量化.md)；[第 80 章](../../book-21-transformer-architecture-evolution/chapters/80-step-3-5-flash的mtp与滑动窗口moe.md) |
| DeepSeek V4.1-Flash；K2 Horizon；Qwen3.8；GLM-5.3-Flash | [第 81 章](../../book-21-transformer-architecture-evolution/chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md)；[第 82 章](../../book-21-transformer-architecture-evolution/chapters/82-k2-horizon-mova-36b-a4b与uno.md)；[第 83 章](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)；[第 84 章](../../book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md) |
| DeepSeek V4 Flash Vision；GLM-5.2；GPT-OSS；Kimi K3 | [第 85 章](../../book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)；[第 86 章](../../book-21-transformer-architecture-evolution/chapters/86-glm-5.2-indexshare-mtp与长轨迹rl.md)；[第 87 章](../../book-21-transformer-architecture-evolution/chapters/87-gpt-oss开放权重moe-harmony与可变推理.md)；[第 88 章](../../book-21-transformer-architecture-evolution/chapters/88-kimi-k3-kda-stable-latentmoe与百万token-agent.md) |
| GLM-5；Kimi K2.6；Qwen3-Omni；Qwen3-VL | [第 89 章](../../book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)；[第 90 章](../../book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)；[第 91 章](../../book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md)；[第 92 章](../../book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md) |
| Qwen3.5-Omni；Kimi K2.7 Code；Gemini 4 Argon | [第 93 章](../../book-21-transformer-architecture-evolution/chapters/93-qwen3.5-omni-aria-aut-timestamp与流式语音.md)；[第 94 章](../../book-21-transformer-architecture-evolution/chapters/94-kimi-k2.7-code-long-horizon-coding-agent.md)；[第 95 章](../../book-21-transformer-architecture-evolution/chapters/95-gemini-4-argon长程推理与网络安全防御.md) |

## 运行时与模型发布专题

架构索引处理模型内部公开结构；接口配置、长任务环境和 harness 有各自的主讲位置。遇到相似的产品字段时，沿下表分别追到请求预算、验证器或评测协议。

| 查询主题 | 深入阅读 |
| --- | --- |
| GPT-6 Astra 的上下文、工具与运行时预算 | [第六册第 18 章](../../book-06-llm-deployment/chapters/18-gpt-6-astra长上下文与工具预算.md) |
| GLM-5.3 的长任务环境、奖励捷径与验证器 | [第十六册第 20 章](../../book-16-reasoning-models/chapters/20-glm-5.3长任务环境与验证器.md) |
| Kimi K3 发布证据与长任务 harness | [第十七册第 15 章](../../book-17-agent-tool-use/chapters/15-kimi-k3发布证据与长任务harness.md) |

## 继续阅读

- 注意力与状态记忆：第二十一册第 10—14、61—77 章。
- MoE 参数、路由与 serving 账本：第二十一册第 21—23、66 章；第六册的推理与部署专题。
- Agent 工具、任务 harness 与运行时状态：第十六册、第十七册和第二十册相应专题。
- 论文与技术报告索引：[`PAPERS.md`](../../PAPERS.md)。
