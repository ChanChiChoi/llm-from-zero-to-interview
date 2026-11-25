# DeepSeek-V4：百万上下文配置与推理实现

> 本章以 DeepSeek-V4 技术报告 v1 的 Pro/Flash preview 为模型级结果对象，使用 Hugging Face 固定 revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 核对公开配置和参考代码。CSA/HCA 的压缩与 query 读取机制见[第 74 章](74-csa-hca压缩注意力.md)；这里聚焦 V4 层表、百万 token 评测、资源基线、异构缓存和具体实现证据。后续 Pro-0813 与 V4.1-Flash 不与 preview 论文数据混用。

## 77.1 先确定模型与来源版本

“DeepSeek-V4”在报告、模型配置和后续 checkpoint 中对应不同证据对象。本章的层配置、Table 7 结果和 FLOPs/KV 估算来自 2026-04-26 的 [DeepSeek-V4 技术报告 v1](https://arxiv.org/html/2606.19348v1)；Pro preview 的配置与参考实现链接锁定在同一个 [Hugging Face revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1)。报告定义模型设计和实验结果，固定配置对应一个可检查的 artifact，参考代码公开一种执行路径。三者都不能单独证明托管 API 或生产 serving 的实际运行形态。

### 77.1.1 V4 preview 报告定义模型规模与百万窗口

技术报告将 Pro 定义为 1.6T 总参数、49B 激活参数，将 Flash 定义为 284B 总参数、13B 激活参数；两个 preview 模型的上下文窗口都是 1M token。参数规模说明 MoE 模型的权重总量和每 token 激活量，窗口则描述可处理序列长度。这些值来自报告中的 preview 模型设置，不能据此推断每个 1M 输入都能被准确检索。

### 77.1.2 固定配置与参考代码是指定版本的实现证据

Pro preview 的固定 [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) 暴露 `num_hidden_layers`、`index_topk`、`sliding_window`、`compress_ratios`、`q_lora_rank` 和 MoE 配置。文件可用于核对论文中公开的架构设置，不能证明服务端必然采用同样的 kernel、精度或并行拓扑。固定 revision 下的 [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py) 与 [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py) 是公开参考代码；源码存在不等于权重已下载或本机推理已复现。

### 77.1.3 Pro-0813 与 preview 需要分开记账

[官方 Pro-0813 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) 将它标为 superseding preview 的正式发布，并说明它沿用 preview 结构、附带 DSpark draft head。它是后续 checkpoint，不能把它的评测结果回填到 preview 论文 Table 7。V4.1-Flash 则有另一套架构，留给[第 81 章](81-deepseek-v4.1-flash-causal-encoder-decoder.md)；API、reasoning effort 产品行为和 Agent harness 结果不属于本章的架构配置主线。

## 77.2 Flash 与 Pro 的逐层配置怎样兑现混合注意力

第 74 章解释了 CSA/HCA 分别怎样压缩和读取历史；这里看它们如何被放进具体模型的层表。两种 preview 模型都采用 $m=4$ 的 CSA、$m'=128$ 的 HCA 和 $128$ token 的局部窗口，但层数、CSA top-k、query heads 与 MoE 容量不同。下表中的配置来自技术报告 §4.2.1；固定 Pro config 可复核对应字段。

| 模型 | 注意力层排布 | CSA 配置 | HCA 配置 | 其他长上下文配置 | MoE 配置 |
| --- | --- | --- | --- | --- | --- |
| V4-Flash | 43 层；前 2 层纯 SWA，后续交错 CSA/HCA；按排布为 21 CSA、20 HCA | $m=4$，top-k $512$ | $m'=128$ | 局部窗口 $128$；$64$ 个 query heads，head dimension $512$，query compression dimension $1024$ | $256$ routed + $1$ shared；每 token 激活 $6$ 个 routed experts |
| V4-Pro | 61 层；前 2 层 HCA，后续交错 CSA/HCA；按排布为 30 CSA、31 HCA | $m=4$，top-k $1024$ | $m'=128$ | 局部窗口 $128$；$128$ 个 query heads，head dimension $512$，query compression dimension $1536$ | $384$ routed + $1$ shared；每 token 激活 $6$ 个 routed experts |

MoE 与注意力层数共同构成完整架构，但它们回答不同问题：MoE 稀疏选择改变每个 token 的 FFN 计算路径，CSA/HCA 改变历史 KV 的存储粒度和 query 读取规模。mHC、MoE 路由和 MTP 的机制分别由[第 78 章](78-mhc双随机残差连接.md)、[第 21 章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)及[第 58 章](58-DeepSeek-V2-V3-R1架构训练.md)承担，本章不把它们当作 CSA/HCA 的子机制。

### 77.2.1 Flash：前两层走纯 SWA，后续交错 CSA 与 HCA

Flash 的 43 个 Transformer 层中，最初 2 层使用纯 Sliding Window Attention，其余层交错使用 CSA 和 HCA。CSA 的压缩率为 $4$，每个 query 的 indexer top-k 上限为 $512$；HCA 的压缩率为 $128$。两类混合注意力层都另有 $128$ token 局部窗口。报告还给出 $64$ 个 query heads、head dimension $512$ 和 query compression dimension $1024$。

### 77.2.2 Pro：两层 HCA 起步，随后交错 CSA 与 HCA

Pro 的 61 个 Transformer 层先放置 2 个 HCA 层，后续交错 CSA/HCA。其 CSA top-k 上限为 $1024$，HCA 压缩率仍为 $128$，局部窗口仍为 $128$；相比 Flash，Pro 的 query heads 增至 $128$，query compression dimension 为 $1536$。固定配置中 `num_hidden_layers=61`、`index_topk=1024`、`sliding_window=128`、`q_lora_rank=1536` 与按层排列的 `compress_ratios` 可与报告相互核对。

### 77.2.3 把压缩比例换算成一百万 token 时的候选规模

在接近配置上限的 $1{,}048{,}576$ 个位置处，按序列维压缩比例估算，CSA 约有 $1{,}048{,}576/4=262{,}144$ 个全局压缩候选；每 query 的主 attention 至多选择 Flash 的 $512$ 个或 Pro 的 $1024$ 个压缩条目。HCA 约有 $1{,}048{,}576/128=8{,}192$ 个压缩条目，每个 query 对所有因果可见的 HCA 条目做密集读取；两种路径还各自读取最近的 $128$ 个原始位置。

这些是序列末端附近的候选条目数估算，忽略压缩块边界和未完成尾部。它们显示了配置如何限制或缩短 query 的读取集合，但不包含 entry width、量化 scale、indexer 打分和 kernel 工作量，不能直接当作实际显存字节、FLOPs 或延迟。

## 77.3 百万 token 窗口和百万 token 检索成绩是两类证据

Pro 固定配置的 `max_position_embeddings=1048576` 对应报告所称的 1M 窗口。这个字段描述模型配置允许的序列长度；它本身不是检索质量指标。报告 Table 7 进一步分别列出 MRCR 1M 与 CorpusQA 1M 的结果，并按 Non-Think、High、Max 三种 reasoning effort 报分。表中的 effort 档是同一系列模型的不同推理模式，不是三个权重 checkpoint。

### 77.3.1 `max_position_embeddings` 描述配置容量

设置一个 million-token position embedding 上限，表明模型和实现为该序列长度预留了位置处理能力。读者仍要问：给定任务需要找什么信息，证据离 query 多远，有多少干扰项，评测判定规则是什么。配置字段回答“能输入多长”，不能替代这些问题。

### 77.3.2 按报告原指标比较 MRCR 1M 与 CorpusQA 1M

DeepSeek-V4 技术报告 v1 Table 7 的长上下文分数如下；分数口径保留报告原名，不将其改写成百分比准确率。

| 模型 / reasoning effort | MRCR 1M (MMR) | CorpusQA 1M (ACC) |
| --- | ---: | ---: |
| Flash / Non-Think | 37.5 | 15.5 |
| Flash / High | 76.9 | 59.3 |
| Flash / Max | 78.7 | 60.5 |
| Pro / Non-Think | 44.7 | 35.6 |
| Pro / High | 83.3 | 56.5 |
| Pro / Max | 83.5 | 62.0 |

这些是作者报告的模型评测值。不同 effort 有不同计算/推理预算，分数差异不能解释为架构层表或 CSA/HCA 的独立因果效应。MRCR 用于上下文检索；论文称 CorpusQA 更接近真实问答场景，但这一表述不等于对所有长文档和真实工作流的验证。

### 77.3.3 128K 之后的下降限制“百万上下文”结论

报告指出，Pro 的 MRCR 检索表现到 $128\text{K}$ 较稳定，超过这一长度后出现下降；在 $1\text{M}$ 处仍能在报告的比较中保持有竞争力的分数。这比“1M context 一定能记住全部内容”更窄，也更有用：容量支持、指定 benchmark 检索、真实任务可靠性是三个不同判断。

## 77.4 资源节省要按基线和精度拆开解释

技术报告分别给出相对 DeepSeek-V3.2 的单 token FLOPs/KV cache 估算，以及相对 BF16 GQA8 baseline 的缓存比例。两种比较的模型、精度和分母都不同，必须各自保留；报告还把混合精度列为效率来源之一，所以不能把全部变化都归因给 CSA/HCA。

### 77.4.1 V3.2 对照是模型级单 token FLOPs 与 KV 估算

在 $1\text{M}$ token 设置下，报告估算 V4-Pro 使用 DeepSeek-V3.2 单 token inference FLOPs 的 $27\%$、KV cache 的 $10\%$；V4-Flash 分别为 $10\%$ 和 $7\%$。FLOPs 按 equivalent FP8 FLOPs 计。这些是作者给出的模型级效率估算，不是同一 GPU 上的 online latency 测试，也不代表 batch、prefill/decode 比例变化后的吞吐。

### 77.4.2 BF16 GQA8 的约 2% 使用另一种缓存基线

报告还以 BF16、GQA8、head dimension $128$ 的注意力配置为缓存 baseline，称 V4 系列在 1M 场景下 KV cache 约为该基线的 $2\%$。它回答的是“相对某个常见 BF16 attention 配置，KV cache 缩到多少”；它与前一节的 V3.2 相对比例使用不同分母，不能组合成同一项节省率。

### 77.4.3 压缩率之外还有混合精度和索引计算

V4 将 KV 中 RoPE 相关维度存为 BF16，其他维度存为 FP8；报告称这种混合表示相较全 BF16 的 KV cache 几乎减半。Lightning Indexer 计算采用 FP4，MoE routed expert weights 也使用 FP4。长上下文效率来自压缩序列长度、稀疏筛选和低精度存储/计算等共同设计；报告并未通过完整的 1M 单因素消融将每项收益精确分账。

## 77.5 异构 KV cache 保存不同更新规则的状态

混合层的 cache 不再是每层相同长度、每 token 固定字节数的一块数组。DeepSeek-V4 报告把 cache 分成 CSA/HCA 的 classical KV 和保存 SWA/未完成压缩 token 的 state cache；Lightning Indexer 还有维度不同的 key。不同状态长度、写入时机和恢复条件不同，必须在模型状态账本中分别理解。

### 77.5.1 主 KV 与 state cache 的条目和更新规则不同

CSA/HCA 的压缩条目在压缩块完成后进入主 KV cache；indexer 保存自己的检索 key；SWA state cache 保留最近的原始 token KV；当前块中还不足以形成压缩条目的 token 和 hidden states 必须留在 tail buffer。随着请求推进，完整块可进入压缩序列，未完成的尾部仍按 token 状态保留。这里描述的是 V4 公开 cache 布局，不是可套用到任意 hybrid 模型的通用实现。

### 77.5.2 LCM 对齐块把两种压缩流放入共同布局

报告用 $\mathrm{lcm}(m,m')$ 个原始 token 作为 CSA/HCA 主 cache 的对齐跨度。V4 取 $m=4$、$m'=128$ 时，$\mathrm{lcm}(4,128)=128$：每个跨度包含 $32$ 个 CSA 压缩条目和 $1$ 个 HCA 压缩条目。这让两类压缩流可以映射进共同的主 cache block；框架的物理 page 大小和分配实现仍可能不同。

### 77.5.3 前缀缓存复用到完整块，窗口状态另行管理

报告的 on-disk cache 可以复用已经完成的 CSA/HCA 压缩条目。若共享前缀末尾落在未完成压缩块中，那一段原 token state 并未作为压缩条目存入磁盘，恢复时仍须重算。SWA 原始 KV 的更新规律不同，报告分别比较完整 SWA 缓存、周期 checkpoint 和不缓存 SWA 的存储—重算取舍。通用前缀缓存的数据结构与引擎实现见[第二十四册第 23 章](../../book-24-llm-inference-engine/chapters/23-prefix-caching与prompt-cache.md)；本章只说明 V4 特有的状态边界。

## 77.6 固定实现和框架案例展示结构之外的执行条件

论文告诉我们设计和报告估算，固定参考代码让读者检查模型实现路径，而框架 [SGLang v0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20) 与 PR 描述某些 GPU 与 workload 下的实测。三类材料的证据层级不同；以下数字均是来源报告值，不是本地运行结果。

### 77.6.1 固定参考实现把配置连接到模型与 kernel 文件

Pro preview 固定 revision 下的 [`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py) 中，attention 按 `compress_ratios[layer_id]` 选择层路径，并为 CSA 路径配置 indexer；[`inference/kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py) 提供 FP4/FP8 GEMM、稀疏 attention 与 Sinkhorn 相关 kernel。它们适合作为“配置如何进入一个公开实现”的代码入口。实现文件存在并不能证明其在某张卡、某个 batch 或服务端 API 中取得相同性能。

### 77.6.2 SWA 分支点缓存的收益绑定共享前缀负载

SGLang v0.5.20 的 unified-tree [PR #34565](https://github.com/sgl-project/sglang/pull/34565) 为共享前缀请求保留 SWA 状态的分叉点。在 DeepSeek-V4-Flash-0731 的一个内部 shared-prefix 模拟中，64 条请求分成 8 组，每组共享 24,576 token 的 system prompt，question 为 8,192 token、输出 128 token。开启窗口外状态释放时，PR 报告 token hit rate 从 $43.81\%$ 升至 $60.75\%$，mean TTFT 从 $1.570\text{s}$ 降至 $1.070\text{s}$；release note 用四舍五入值 $43.8\%\to60.8\%$ 与 $1.57\text{s}\to1.07\text{s}$。

这个案例说明压缩 KV 命中不保证 SWA 分支状态也能恢复，cache 结构要保留正确的 branch point。它是 SGLang PR 作者在指定请求分组和共享 prompt 上的结果，不是本地复现，也不代表任意生产流量的 hit rate 或 TTFT。

### 77.6.3 B200 kernel 与 SM120 serving 数字不能合并成一个加速倍数

SGLang v0.5.20 release 报告，B200/SM100-103 上的 TRT-LLM CSA/HCA attention kernel 相对 FlashMLA，prefill 约 $1.2\times$、decode 约 $1.45\times$；对应 [PR #30805](https://github.com/sgl-project/sglang/pull/30805) 的 FP8、TP=1 kernel 级比较，不是模型端到端延迟。

另一条 [PR #29927](https://github.com/sgl-project/sglang/pull/29927) 针对 4× RTX PRO 6000（SM120）上的 V4-Flash serving 路径，发布说明报告 batch 1、8K 输入的 TPOT 从 $36.1\text{ms}$ 降至 $10.5\text{ms}$，并报告从 8K 到 128K 输入时 TTFT 约下降 $20\%$。该改动包括 paged-MQA indexer、FP4 MoE backend 和 page-split 等多项优化。硬件、baseline、指标和实现范围均与 B200 kernel 比较不同，不能把两组数字相乘或写成 V4 的通用加速率。

## 77.7 用四种证据分别回答四个问题

读到“支持百万上下文”时，先分别问配置允许多长、指定任务上检索如何、相对哪个 baseline 节省资源、哪个引擎和 workload 下测得什么延迟。V4 的公开证据说明：一套具体的 CSA/HCA 层表、局部窗口、低精度、缓存布局和服务内核共同支撑了长上下文；报告没有把百万窗口、任意任务的质量保证和通用线上速度合成一个结论。

### 77.7.1 配置、质量、资源和延迟不能互相代替

最大位置数是配置事实；MRCR/CorpusQA 是模型在指定评测条件下的质量结果；FLOPs/KV 比例是相对基线的报告估算；SGLang PR 数字是指定硬件和负载的框架性能报告。做横向比较时先对齐模型版本、baseline、指标、精度、硬件与 workload。不能用 context 字段代替检索结果，不能用 cache 比例代替 latency，也不能把 kernel 倍数写成模型能力提升。

### 77.7.2 将机制、服务和版本问题交回相应章节

CSA/HCA 的压缩与读取机制回到[第 74 章](74-csa-hca压缩注意力.md)；mHC、MoE 和 V4.1-Flash 分别见[第 78 章](78-mhc双随机残差连接.md)、[第 21 章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)和[第 81 章](81-deepseek-v4.1-flash-causal-encoder-decoder.md)。KV cache 账本可接读[第六册第 3 章](../../book-06-llm-deployment/chapters/03-kv-cache与内存管理.md)，PagedAttention 和引擎性能测试可接读[第二十四册第 17 章](../../book-24-llm-inference-engine/chapters/17-pagedattention的核心思想.md)与[第 53 章](../../book-24-llm-inference-engine/chapters/53-构建serving-engine的benchmark指标和调参实验框架.md)。
