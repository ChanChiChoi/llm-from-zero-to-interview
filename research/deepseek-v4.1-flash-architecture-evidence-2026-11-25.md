# DeepSeek-V4.1-Flash 架构、缓存与版本证据

研究日期：2026-11-25。范围：为第二十一册第 81 章核对 DeepSeek-V4.1-Flash 的 CED、CSA2、Hierarchical Sparse Indexer、SWA Bounded Replay、FP4 global KV、Engram、Single-Pass mHC、DSpark 和视觉输入路径；同时确认 2026-10-17 的 1017 权重更新怎样影响当前模型身份。目标读者已理解 causal Transformer、KV cache、MoE 和注意力压缩基本概念。本条只保存外部发现与证据边界；章节职责和编辑取舍记录在 `progress/`。

本轮检索词：`DeepSeek-V4.1-Flash technical report CED CSA2`、`DeepSeek-V4.1-Flash-1017 config.json architecture`、`DeepSeek V4.1 Flash bounded replay SWA cache`、`DeepSeek CSA2 Full Reindex Reuse hierarchical sparse indexer`、`DeepSeek V4.1 Flash KV cache lecture notes explained`。首轮并行查 DeepSeek 官方技术报告、两代官方模型卡/配置、官方 API changelog，以及面向读者的工程解释文章和课程 notes；公式、配置和效果数回到报告/固定配置核验。

## 模型版本和参数口径

| 发现 | 来源支持 | 边界 |
|---|---|---|
| DeepSeek-V4.1-Flash 于 2026-09-10 发布；2026-10-17 官方发布 V4.1-Flash-1017 更新，并称 API 调用方式和价格不变。 | DeepSeek [API changelog](https://api-docs.deepseek.com/updates/)，2026-09-10 发布项和 2026-10-17 更新项。 | API alias 指向服务，不是固定权重标识；不把 API 行为当成 checkpoint 架构。 |
| 原始技术报告 arXiv:2609.19969 v1 描述 2026-09 发布模型；报告的结构、实验和部署结果以该版本为来源对象。 | DeepSeek-AI [技术报告 v1](https://arxiv.org/html/2609.19969v1)，2026-09-17，§§1–5。 | 报告未重做 1017 的逐组件架构消融或服务性能测量。 |
| 原始发布配置固定 revision `dba1be0a40aa45a94ad051997016db3960a90277`；可读 `inference/config.json` 给出 40 层、窗口 128、CSA2 压缩率、top-k、候选池、Engram 和 DSpark 等字段。 | DeepSeek 官方 [固定 revision 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/config.json)。 | 这是 0909 发布 checkpoint 的配置快照，不代表托管 API 的具体服务部署或内核。 |
| 1017 官方模型卡称其为 V4.1-Flash 的 minor update，并发布新的 benchmark 表；官方 changelog 说明 API/Web/APP 已更新到 1017。 | DeepSeek 官方 [V4.1-Flash-1017 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)；[changelog](https://api-docs.deepseek.com/updates/)。 | 分数是发布方结果，按指定 benchmark/harness 条件理解；不是架构单因素证据。 |
| 1017 的公开 `config.json`（release commit `1bae0e1088993b9b82de08f59de5be5a34e42170`）与原始固定配置在本章使用的结构字段上匹配：40 层、`compress_ratios` 的 SWA/m=2/m=1 分段、384 routed + 1 shared expert、top-6、window 128、index top-512、候选池 2048×8、Engram 层和 DSpark 形状。 | DeepSeek 官方 [1017 固定 release 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)，对照上条原始 revision 配置。 | 只据公开配置断言这些结构字段相符；不推断完整权重完全相同、训练数据/后训练未变或所有 runtime 条件相同。vLLM 官方 recipes 也将 1017 作为权重更新单独登记，但这是推理项目的集成记录，不代替 DeepSeek 模型声明。 |
| 模型卡正文分别给出 552B backbone 参数和 196B Engram 参数；Hugging Face 文件页的自动统计显示 763B params。 | 两代 DeepSeek 模型卡正文；Hugging Face 文件页自动统计。 | 发布资料没有解释这些口径如何对账；正文分开引用，不推算总量或显存。 |

## CED、CSA2 和索引范围

### Causal Encoder-Decoder（CED）

- 报告 §2.2 将 40 层分成 20 层 causal encoder 和 20 层 decoder。对 global attention，decoder 各层从 encoder 最后一层 hidden state 通过各自的投影产生 global KV 及 compression weights；这些不是由 decoder 自己的 hidden states 逐层产生。
- 每层的 SWA K/V 仍从该层自己的 hidden state 产生。对 $N\gg n_{\mathrm{win}}$ 的长序列，报告把 prefill 近似写为 $O(NL/2+n_{\mathrm{win}}L/2)\approx O(NL/2)$；这是模型报告的计算近似，不是任意硬件上延迟必然减半。局部 SWA 的回放项对短轮次和缓存命中路径仍重要。
- 报告说 CED 受 YOCO 启发；两者不能混为同一架构。报告将 CED 的差异写成保留层级 SWA 计算并使用 layer-dependent projection。

### Compressed Sparse Attention 2（CSA2）

- 原 V4 的 CSA/HCA hybrid 与 V4.1-Flash 的 pure CSA2 区分清楚。CSA2 每个 attention layer 仍有自己的 global Q 和 SWA KV；差异在 main KV、indexer K 与 Top-K indices 的生成/复用。
- **Full** 生成 main KV、indexer Q/K 并计算新 Top-K；**Reindex** 复用最近 Full 提供的 main KV 与 indexer K，用本层 query 重排并产生新 Top-K；**Reuse** 复用 main KV 与最近一次 Full/Reindex 的 Top-K，不重新运行 indexer。
- 报告 §2.3 指出 CSA2 移除 V4 CSA 压缩条目间的源重叠与绝对位置嵌入，并从 main KV 投影 indexer K；因此不应将 V4 CSA 压缩构造未经核验地搬到 CSA2。
- 报告的布局是 encoder 两层 SWA-only，随后 18 层使用 $m=2$，三组 `Full + 5 Reuse`；decoder 20 层使用 $m=1$，一组 `Full + 3 Reuse`，之后四组 `Reindex + 3 Reuse`。层排布来自报告设置，并与固定配置字段交叉检查，不从 ratio 字段单独猜 mode。

### Hierarchical Sparse Indexer（HSI）

- HSI 只用于 CED decoder。第一个 Full layer 扫描当前 query 的因果可见 main-KV 范围，并以每个 block 内最大 index score 选择候选 block；报告示例为 2,048 个 block × 每块 8 个位置，即至多 16,384 个候选位置。
- 后续 Reindex layer 只在候选池内重新评分并选择自身 Top-512。候选池固定时，后续索引评分的每 query 成本不再随完整上下文长度线性增长；第一个 Full layer 仍执行全可见范围扫描。该结论不等于整个模型或整个 decode 成本与上下文长度无关。
- 候选池或最终 Top-K 可能漏掉相关证据；报告没有给出可普遍外推的候选召回保证。任何召回乘积公式只能是明示独立假设下的教学模型，不能代替实测。

## SWA Bounded Replay 与 FP4 cache

- V4.1-Flash 每层保留 128-token SWA。缺失局部 KV 时，exact reconstruction 涉及跨层 SWA 依赖，报告称精确恢复需重放约 $L\times n_{\mathrm{win}}$ tokens；Bounded Replay 只重放最近 $n_{\mathrm{win}}$ tokens，并截断回放起点之前的 SWA 状态，因此是近似恢复，不保证与完整前向的状态/hidden/logits 完全相同。
- 报告区分 encoder replay 与 decoder replay：前者在重放缓存前缀时复用已缓存 global KV、只重建 SWA 状态；后者在 prefill 尾部重新经过 decoder 层生成供后续 decode 使用的局部状态，不将这段 replay 结果用作 prefix cache。作者报告对回复质量影响可忽略，并称 post-training 也模拟该路径；这是作者在该模型/实验设置下的观察。
- 报告的部署方案将短 TTL SWA KV 放入每机器 host DRAM 池，global KV 留在持久 cache；这属于报告描述的系统策略，不是 CSA2 数学定义或所有引擎默认行为。
- main KV 采用 E2M1 FP4，每 16 channels 共享一个 E4M3 scale；报告选择在 RoPE 后量化，SWA KV 保持 FP8，并在 attention 前反量化。精度收益/代价、block layout 和目标设备实现不能从“FP4”三个字符推断。
- 报告将 890 bytes/token 定义为 global KV cache footprint，约为 V4-Flash 对应值的 1/4；Bounded Replay 与 global cache 缩减共同使报告中的 persistent KV footprint 约为 V4-Flash 的 1/8。两项都是同长度模型/部署比较的发布方指标，不等于权重、SWA 临时状态、索引器、batch workspace 或完整机器内存。
- 报告引言与 §2.4.4 对 FP4 的时序措辞粒度不同：详细小节指出 main-KV QAT 在 post-training 加入；正文依详细小节表述，不声称 45T token 预训练从一开始就用 FP4 cache。

## 其他结构组件与证据边界

- **MoE：** 1 shared expert、384 routed experts、每 token 选择 6 routed experts；总参数/激活参数/通信不能以 top-k 数替代。报告还为图像和文本 token 维护独立的无辅助损失负载平衡 correction bias。
- **Engram：** 报告描述为 token-based conditional memory，196B 参数分配给两个模块，放置在 zero-indexed layers 1 和 14，使用 2/3/4-gram、多头 hashing 与条件门控；它既不是当前请求的 KV 状态，也不是可独立更新的外部 RAG 库。公开资料不足以重建完整 hashing/runtime 行为。
- **Single-Pass mHC：** 对原始 mHC，input-mixing 使用当前层由残差状态产生的 $A_l$，与同次融合产生系数存在依赖；部署版 Single-Pass 使用前一层 $A_{l-1}$ 消除该依赖。报告称 Mega-mHC 将激活流量从原实现约 $(4n+4)d$ 降到 $(2n+2)d$，属发布方 kernel/系统测量；第 78 章承担 mHC 核心数学。
- **DSpark：** 报告描述三层 drafter、128-token sliding window、单次前向并行产生五个 draft positions、Markov head 与逐位置 confidence head，并依 engine throughput 动态选择 verification length。DSpark 在 backbone pretraining 后单独训练；这与第 80 章 Step 3.5 的 MTP-3 训练/推理路径不同。配置或类注册不能证明接受长度、端到端加速或目标硬件 SLO。
- **视觉：** DeepSeek-ViT 由头训练，使用 2D-RoPE、线性 patch embedding、RMSNorm/SwiGLU；3×3 pixel-unshuffle 把空间 token 数约降为 1/9，MLP projector 后将 image embeddings 放到对应图像 token 位置，与文本共同通过语言主干。模型原生视觉输入不表示具有 API、外部媒体或工具权限。
- **效果归因：** 45T 多模态 token、64K 序列长度下从头训练 sparse attention、处理到 34T token 时扩展到 1M 是该模型报告的训练设置。作者自报的 benchmark、KV/FLOPs 估算和部署数字不拆分单个架构改动的因果贡献；没有本地完整权重、GPU、kernel 或 benchmark 复现。
- **1017 更新：** 1017 官方模型卡报告若干更新成绩，并注明部分工具 benchmark 使用指定 harness。其结构配置与原始配置中本章关注字段相符；报告结果仍不能单独证明架构改动或部署加速。

## 教学来源和使用范围

| 来源 | 用途 | 使用边界 |
|---|---|---|
| TRAIL Labs, [DeepSeek V4.1 Flash: smaller KV cache, same post-training](https://www.traillabs.ai/en/blog/deepseek-v4-1-flash-tech-report) | 将 CED、层间 KV 复用、FP4 与 SWA replay 放在不同 cache 轴上解释；用于审查章节讲法和概念分组。 | 博文作者明言未复现模型；数值与机制回查 DeepSeek 报告。 |
| OrigoTech, [DeepSeek V4.1-Flash explained: KV cache, 1017, Harness](https://origotech.dev/articles/deepseek-v4-1-flash-technical-deep-dive-kv-cache/) | 辅助识别 0909 与 1017 版本时间线，梳理 Full/Reindex/Reuse 讲解次序。 | 文章汇总性质，不作架构/性能主证据。 |
| Stanford MS&E 319, [Lecture 5: KV Caching, Sparse Attention, and Compression](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf) | 对照 KV cache、压缩条目和稀疏选择的课程解释层次。 | 该讲义以一般概念和 V4 为例，不是 V4.1 配置或效果证据。 |

## 1017 更新成绩与评测条件

- 官方 1017 模型卡列出 0909 与 1017 的对照：Terminal-Bench 4.0 为 31.2→42.3，DeepSWE v1.1 为 74.2→77.0，ProgramBench 为 20.3→29.1；另列 HLE、Automation-Bench、Agents’ Last Exam、Apex Bench 与视觉 benchmark。来源：[1017 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。
- 官方 changelog 说明 HLE 使用纯文本子集，Terminal-Bench 4.0 使用 Claude Code framework，DeepSWE 使用 Mini-SWE-V2 framework，ZeroBench-main 使用 Pass@5；这些分数是发布方在指定评测条件下的报告。来源：[DeepSeek 更新日志](https://api-docs.deepseek.com/updates/)。
- 官方 changelog 的 2026-11-13 条目介绍当期 API 产品功能，没有公布新的公开 checkpoint；当前 API 定价页仍列出 DeepSeek-V4.1-Flash-1017 版本。产品功能更新不能替代模型结构或权重版本证据。来源：[更新日志](https://api-docs.deepseek.com/updates/)与[模型定价页](https://api-docs.deepseek.com/quick_start/pricing/)。
- 这些整体 benchmark 对照没有提供 CED、CSA2、FP4、Engram 或 DSpark 的逐组件消融，不能归因某一架构机制。
