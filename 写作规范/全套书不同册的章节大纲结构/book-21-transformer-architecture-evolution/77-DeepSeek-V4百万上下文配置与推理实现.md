# 第 77 章 DeepSeek-V4：百万上下文配置与推理实现

本章承接第 74 章的 CSA/HCA 机制，回答这些机制在 DeepSeek-V4 Pro 与 Flash 中怎样形成具体层表、公开的百万 token 结果和异构 KV cache。主线是把模型配置、长上下文质量评测、相对不同基线的资源估算、官方参考实现和框架性能报告拆开阅读；其中任何一项都不能单独推出“任意百万 token 输入都可准确检索”或“所有硬件都能获得同样延迟”。配置和结果以 DeepSeek-V4 技术报告 v1 的 preview 模型为准，固定 revision 用于检查字段和参考代码；Pro-0813 与 V4.1-Flash 的后续版本证据不混入 preview 结果。第 74 章完整讲压缩与读取机制，第 78 章讲 mHC，第 6、24 册负责推理服务和引擎细节。

## 77.1 先确定模型与来源版本

本节划清本章的数据对象和证据层级：DeepSeek-V4 报告中的 preview Pro/Flash 是模型级质量与资源估算的对象；固定 Hugging Face revision 用来核对某个时点的配置和公开参考实现；后续 Pro-0813 是在 preview 结构上增加 DSpark 的正式 checkpoint。读者据此避免把 API 名称、后续 checkpoint 或框架测试数字倒灌到论文原始对照中。

### 77.1.1 V4 preview 报告定义模型规模与百万窗口

DeepSeek-V4 技术报告 v1 报告 Pro 为 1.6T 总参数、49B 激活参数，Flash 为 284B 总参数、13B 激活参数，两者的配置窗口均为 1M token。本节说明后文模型层表、Table 7 的 1M 评测和相对 V3.2 的效率数字均对应报告中的 preview 模型；“总参数”“激活参数”和“上下文上限”各自描述不同对象。

### 77.1.2 固定配置与参考代码是指定版本的实现证据

固定 Hugging Face revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 暴露 Pro preview 的 `config.json`、`inference/model.py` 与 `inference/kernel.py`。配置字段可核对论文中的层数、压缩比例、index top-k、窗口及精度；参考代码显示公开实现路径。它们不能单独证明托管 API 或生产 serving 使用同一配置和内核，也不等于已经下载权重或本地复现。

### 77.1.3 Pro-0813 与 preview 需要分开记账

官方 Pro-0813 模型卡将它标为 superseding preview 的正式发布，并说明其沿用 preview 结构、附带 DSpark draft head。该版本报告的 benchmark 与 preview 论文 Table 7 不是同一 checkpoint 的数据；本章只将此版本信息作为边界提示，不展开 API、effort 产品行为或 Agent harness 评测。V4.1-Flash 使用另一套架构，留给第 81 章。

## 77.2 Flash 与 Pro 的逐层配置怎样兑现混合注意力

本节从报告的模型设置和 Pro 固定 config 中读取 CSA、HCA、纯 SWA 的逐层排布，并把层类型与参数配置对齐。两种模型都使用 CSA 压缩率 $m=4$、HCA 压缩率 $m'=128$ 和 $128$ token 的局部窗口，但层数、头数、query 压缩维度和 CSA top-k 不同；这些是 V4 具体配置，不是 CSA/HCA 的通用常数。

### 77.2.1 Flash：前两层走纯 SWA，后续交错 CSA 与 HCA

报告为 Flash 配置 43 个 Transformer 层，前两层使用纯 Sliding Window Attention，其后交错使用 CSA/HCA。CSA 的 $m=4$、indexer top-k 为 $512$；HCA 的 $m'=128$；两者的局部窗口为 $128$。报告给出 $64$ 个 query heads、attention head dimension $512$、query 压缩维度 $1024$，以及每个 MoE 层 $256$ 个 routed experts 加 $1$ 个 shared expert、每 token 激活 $6$ 个 routed experts。重点是读懂该模型的完整配置组合，而不是复述两种机制的公式。

### 77.2.2 Pro：两层 HCA 起步，随后交错 CSA 与 HCA

报告为 Pro 配置 61 个 Transformer 层，前两层使用 HCA，其后交错 CSA/HCA。CSA 的 $m=4$、indexer top-k 为 $1024$；HCA 的 $m'=128$；局部窗口仍为 $128$。配置还包括 $128$ 个 query heads、head dimension $512$、query 压缩维度 $1536$，以及 $384$ routed experts 加 $1$ 个 shared expert、每 token 激活 $6$ 个 routed experts。固定 config 中的 `num_hidden_layers`、`index_topk`、`sliding_window`、`compress_ratios` 和 MoE 字段可与报告相互核对；MoE、mHC 与 MTP 的机制不在本章重复主讲。

### 77.2.3 把压缩比例换算成一百万 token 时的候选规模

在接近 $1{,}048{,}576$ 个上下文位置的序列末端，$m=4$ 对应约 $262{,}144$ 个 CSA 压缩候选，主注意力每 query 的 top-k 上限分别是 Flash $512$、Pro $1024$；$m'=128$ 对应约 $8{,}192$ 个 HCA 压缩条目，供 HCA 在压缩域密集读取，另有 $128$ 个近期原始位置。这里是依据公开比例推算的候选条目规模，忽略块边界和未完成尾部，不能直接换算成实际显存字节、FLOPs 或延迟。

## 77.3 百万 token 窗口和百万 token 检索成绩是两类证据

本节把配置字段中的最大位置数与任务实测分开，按技术报告 Table 7 读取 MRCR 1M 与 CorpusQA 1M 的指标。报告称 MRCR 用于上下文检索，并指出检索表现到 $128\text{K}$ 较稳定、之后出现下降；CorpusQA 更接近真实场景。模型支持的最大位置数不代表该长度下每个位置、每类问题都能等质访问。

### 77.3.1 `max_position_embeddings` 描述配置容量

Pro 固定配置的 `max_position_embeddings=1048576` 对应报告所说的 1M 窗口。它描述模型配置所允许处理的序列长度，不是“记住全部内容”的结果指标，也不能代替不同距离、证据密度和干扰条件下的检索评估。

### 77.3.2 按报告原指标比较 MRCR 1M 与 CorpusQA 1M

报告 Table 7 分别列出 Non-Think、High、Max 三种 reasoning effort 下的 Flash/Pro 分数。本节保留 `MRCR 1M (MMR)` 与 `CorpusQA 1M (ACC)` 的指标口径，逐列展示结果；不将 effort 模式当作不同权重 checkpoint，也不把单个模型的 Table 7 结果解释成 CSA 或 HCA 的单因素消融。

### 77.3.3 128K 之后的下降限制“百万上下文”结论

报告的 MRCR 分析称性能在 $128\text{K}$ 以内相对稳定，超过该长度后出现下降，但 1M 点仍在其比较中保持可观的检索表现。MRCR 与 CorpusQA 任务定义不同，论文作者对 CorpusQA 的“更接近真实场景”是对评测性质的描述，不是所有实际长文档、代码库或 Agent 工作流都已被验证。

## 77.4 资源节省要按基线和精度拆开解释

本节分别读取报告相对 DeepSeek-V3.2 的单 token 推理 FLOPs/KV cache 估算，以及相对 BF16 GQA8、head dimension 128 的 KV 基线。报告还将 KV 混合精度、indexer 低精度和专家权重低精度与注意力结构共同列为效率来源，因此不能将全部比例归因于 CSA/HCA。

### 77.4.1 V3.2 对照是模型级单 token FLOPs 与 KV 估算

在 1M 上下文设置，报告估算 Pro 使用 V3.2 单 token inference FLOPs 的 $27\%$ 和 KV cache 的 $10\%$；Flash 分别为 $10\%$ 和 $7\%$。FLOPs 按等效 FP8 FLOPs 计，属于报告的模型级估算，不是同硬件在线延迟或吞吐实测，也不代表所有 batch 和请求长度。

### 77.4.2 BF16 GQA8 的约 2% 使用另一种缓存基线

报告另以 BF16、GQA8、head dimension 128 作为常见注意力缓存基线，称 V4 系列在 1M 设置下 KV cache 约为该基线的 $2\%$。这个数字不能与“相对 V3.2 的 Pro 10% / Flash 7%”合并，因为分母、精度与配置不同。

### 77.4.3 压缩率之外还有混合精度和索引计算

论文报告 KV 的 RoPE 维保留 BF16、其余维度使用 FP8，称缓存较全 BF16 几乎减半；Lightning Indexer 的计算采用 FP4，路由专家权重也采用 FP4。正文将序列压缩、索引计算和量化存储分列说明，指出它们分别作用于条目数量、查询筛选和每元素字节/计算，不把联合效率数值误写成单一机制的收益。

## 77.5 异构 KV cache 保存不同更新规则的状态

本节依据报告 §3.5 的 KV cache 设计，说明混合层使传统统一每 token cache 账本不再适用。主 KV cache 保存 CSA/HCA 压缩条目；state cache 保存 SWA 最新窗口和尚未组成完整压缩块的尾部状态；indexer 还需要独立维度的检索键。这里只解释 V4 的状态组成与块边界，不展开通用请求调度或缓存淘汰手册。

### 77.5.1 主 KV 与 state cache 的条目和更新规则不同

CSA/HCA 压缩条目、indexer key、SWA 原始 KV 与未完成压缩尾部具有不同维度、长度和更新条件。请求推进时，完整压缩块进入主缓存；尾部 token 在块完成前仍须保留，SWA 则只维护最近窗口。读者应能区分长期压缩表示和依赖当前位置的有界状态。

### 77.5.2 LCM 对齐块把两种压缩流放入共同布局

报告以 $\operatorname{lcm}(m,m')$ 个原始 token 作为可共享的物理块跨度。V4 配置 $m=4,m'=128$ 时，跨度为 $128$ 个原始 token，对应 $32$ 个 CSA 条目和 $1$ 个 HCA 条目；块对齐有助于同时服务两种压缩流。此布局关系不代表所有框架都采用相同页面大小或分配策略。

### 77.5.3 前缀缓存复用到完整块，窗口状态另行管理

论文的磁盘缓存设计可以直接复用完整 CSA/HCA 压缩条目；前缀最后一个未完成压缩块的原 token 状态仍需重算恢复。SWA 体积和更新规则不同，报告分别讨论完整缓存、周期 checkpoint 和零 SWA 缓存三种存储/重算取舍。通过这一案例说明“共享前缀命中”取决于恢复所需的状态是否齐备，而不只是压缩 KV 是否存在。

## 77.6 固定实现和框架案例展示结构之外的执行条件

本节以固定参考代码和 SGLang v0.5.20 为不同粒度的实现证据。参考代码展示公开模型/内核文件，SGLang 的 PR 与 release 报告指定工作负载和硬件结果；这些材料可说明实现路径和性能敏感点，但来源报告不能替代本地复现或生产验收。

### 77.6.1 固定参考实现把配置连接到模型与 kernel 文件

在 Pro revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 下，`inference/model.py` 负责公开模型执行结构，`inference/kernel.py` 提供 TileLang 等低精度与注意力 kernel。读者可以沿 `compress_ratios`、indexer、局部窗口和 kernel 调用定位配置如何进入推理；文件存在只能证明参考实现公开，不证明对应权重、硬件或端到端 1M 运行已在本环境验证。

### 77.6.2 SWA 分支点缓存的收益绑定共享前缀负载

SGLang v0.5.20 的 unified-tree PR #34565 在 DeepSeek-V4-Flash-0731 的共享 system prompt 负载中保留请求分叉点的 SWA 状态；发布说明报告 token hit rate 从 $43.8\%$ 到 $60.8\%$、mean TTFT 从 $1.57\text{s}$ 到 $1.07\text{s}$。该结果来自特定 shared-prefix workload，说明 SWA 状态恢复会影响前缀复用，不能外推为任意流量的缓存命中或延迟收益。

### 77.6.3 B200 kernel 与 SM120 serving 数字不能合并成一个加速倍数

SGLang v0.5.20 的 B200/SM100-103 PR #30805 报告 TRT-LLM CSA/HCA kernel 相对 FlashMLA 的 prefill 约 $1.2\times$、decode 约 $1.45\times$，这是 FP8/TP=1 的 kernel 级比较。另一条 4× RTX PRO 6000/SM120 PR #29927 报告 batch 1 的 TPOT 在 8K 输入从 $36.1\text{ms}$ 降至 $10.5\text{ms}$，并报告 8K—128K 的 TTFT 改善；该路径组合了 paged-MQA indexer、FP4 MoE 和 page-split 等优化。硬件、baseline、层级和负载不同，不能把这两组数字相乘或称作模型级通用加速。

## 77.7 用四种证据分别回答四个问题

本节汇总章节结论：配置回答模型怎样排层，长上下文 benchmark 回答指定任务上表现怎样，资源估算回答相对明确基线的理论/报告账本，代码与框架 PR 回答指定实现和工作负载下怎样运行。四类证据互有关联，却不能互相代替；百万上下文可用性来自完整模型、精度和推理实现的组合，单一压缩算子不承担全部因果解释。

### 77.7.1 配置、质量、资源和延迟不能互相代替

最大窗口、MRCR/CorpusQA 分数、KV cache 字节、估算 FLOPs、kernel 延迟和共享前缀 TTFT 是不同指标。比较模型或复核主张时应先确认 version、baseline、指标、硬件和 workload，再决定结论能否跨到其他设置。

### 77.7.2 将机制、服务和版本问题交回相应章节

本章只用第 74 章机制解释支撑具体配置，不重讲压缩算子；mHC、MoE 与 V4.1-Flash 的机制分别由第 78、21 和 81 章承接；更完整的 serving 与推理成本回到第 6、24 册。本章最终结论限定为：V4 的公开证据展示了一套特定的百万上下文架构与实现方案，其成绩和成本必须按相应版本与比较条件理解。
