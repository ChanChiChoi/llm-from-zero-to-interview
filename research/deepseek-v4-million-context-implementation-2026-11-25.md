# DeepSeek-V4 百万上下文配置与推理证据

研究日期：2026-11-25。问题范围：如何将 DeepSeek-V4 Pro/Flash 的百万 token 主张还原为指定版本的层配置、质量评测、资源基线、异构 KV cache 与公开推理实现；如何区分论文估算、配置/源码证据和框架自报性能。CSA/HCA 压缩与 query 读取机制见[CSA/HCA 机制研究记录](csa-hca-mechanism-evidence-2026-11-24.md)。

本轮检索词：`DeepSeek-V4 technical report 1M MRCR CorpusQA config layer schedule`、`DeepSeek-V4 Pro compress_ratios index_topk config b5968e9`、`DeepSeek-V4 official Pro-0813 supersedes preview DSpark`、`SGLang v0.5.20 DeepSeek-V4 SWA branching-point cache B200 SM120`、`DeepSeek-V4 million context configuration lecture notes explained`。检索并阅读作者技术报告、官方模型配置/卡片、固定标签的框架 release 与 PR；另查 Hugging Face 技术博客、Berkeley EECS 课程 slides 和 NVIDIA 技术博客，用于检查术语组织、配置解释和工程呈现。教学材料用于形成解释路径，不替代原始结果。

## 模型版本与逐层配置

| 发现 | 来源支持 | 边界 |
| --- | --- | --- |
| V4 preview 包括 Pro（1.6T 总参数、49B 激活）与 Flash（284B 总参数、13B 激活），均为 1M token 上下文配置。 | DeepSeek-AI [DeepSeek-V4 技术报告 v1](https://arxiv.org/html/2606.19348v1)，2026-04-26，Abstract；§4.2.1。 | 报告中的 preview checkpoint，不等于后续 Pro-0813 或 V4.1-Flash。模型卡和报告使用舍入参数量。 |
| Flash 有 43 层，前两层为纯 SWA，其余 CSA/HCA 交错；$m=4$、$m'=128$、CSA top-k 512、局部窗口 128、64 个 query heads、head dimension 512、query compression dimension 1024。MoE 为 256 routed + 1 shared、每 token 激活 6 个 routed experts；mHC expansion 4、Sinkhorn 20 轮、MTP depth 1。 | 同上，§4.2.1 “Model Setups”。 | 精确配置是 V4-Flash preview 报告值，不是所有 Flash 更新版的默认值。 |
| Pro 有 61 层，前两层为 HCA，之后 CSA/HCA 交错；$m=4$、$m'=128$、CSA top-k 1024、局部窗口 128、128 个 query heads、head dimension 512、query compression dimension 1536。MoE 为 384 routed + 1 shared、每 token 激活 6 个 routed experts；mHC expansion 4、Sinkhorn 20 轮、MTP depth 1。 | 同上，§4.2.1 “Model Setups”。 | 论文明确排布；由“前两层 HCA + 其余交错”可得 30 CSA/31 HCA。该计数只适用于该 preview Pro 层表。 |
| Pro 固定 config revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 暴露 `num_hidden_layers=61`、`index_topk=1024`、`sliding_window=128`、`q_lora_rank=1536`、`qk_rope_head_dim=64`、`compress_ratios`、384 routed/1 shared experts、top-6、FP8 quantization config、FP4 expert dtype、`max_position_embeddings=1048576`、mHC `hc_mult=4`/20 Sinkhorn iterations。 | DeepSeek 官方 [Pro preview 固定 config revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json)。 | 这是一个公开 artifact 的字段快照，说明配置披露，不证明线上 API 运行时使用相同 kernel、dtype 或 serving 设置。 |
| Pro-0813 官方模型卡称其 supersedes preview，沿用 preview 结构并附加 DSpark speculative-decoding module。 | DeepSeek 官方 [DeepSeek-V4-Pro-0813 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)，“Introduction”。 | 后续 checkpoint 的 benchmark 不与 preview 论文 Table 7 混合；不在本条记录 API 或 Agent harness 表现。 |

## 百万 token 质量评测与资源基线

### 长上下文评测

DeepSeek-V4 技术报告 v1 Table 7 分别报告 Flash/Pro 的 Non-Think、High、Max：

| 模型/模式 | MRCR 1M (MMR) | CorpusQA 1M (ACC) |
| --- | ---: | ---: |
| Flash / Non-Think | 37.5 | 15.5 |
| Flash / High | 76.9 | 59.3 |
| Flash / Max | 78.7 | 60.5 |
| Pro / Non-Think | 44.7 | 35.6 |
| Pro / High | 83.3 | 56.5 |
| Pro / Max | 83.5 | 62.0 |

来源：DeepSeek-AI [技术报告 v1](https://arxiv.org/html/2606.19348v1)，§5 长上下文评测、Table 7（页面行 808–833）。该表比较报告中的模型/effort 设置，不是 CSA/HCA 层表的单因素消融。报告 §5 说明 MRCR 是 in-context retrieval；MRCR 表现到 128K 较稳定，之后出现下降；作者称 CorpusQA 更接近真实场景。该评述不能推广为对所有真实文件、Agent 任务或证据分布的验证。

### 效率与缓存

- 相对 DeepSeek-V3.2，在 1M-token 场景，报告估算 Pro 的单 token inference FLOPs 为 V3.2 的 27%、KV cache 为 10%；Flash 分别为 10%、7%。FLOPs 使用 equivalent FP8 FLOPs 口径，来自报告 Figure 1/§1，不是在线服务 latency benchmark。
- 另一基线是 BF16 GQA8、head dimension 128。报告称 V4 series 1M 场景下 KV cache 约为该 baseline 的 2%。不能与相对 V3.2 的 10%/7% 混作同一分母。
- §2.3.4 报告称 KV 的 RoPE 维使用 BF16、其他维度用 FP8，较纯 BF16 存储近乎减半；Lightning Indexer 计算用 FP4，MoE routed expert parameters 用 FP4。效率主张包含压缩比例与精度选择等多种因素。
- §3.5.1 将混合 KV cache 分成 CSA/HCA classical KV 与 SWA/未完成压缩 token 的 state cache；indexer 还有独立维度。普通缓存块跨度按 $\operatorname{lcm}(m,m')$ 原始 token 对齐。$m=4,m'=128$ 时一块为 128 原 token，含 32 个 CSA compressed entries 和 1 个 HCA compressed entry；这是按报告公式换算出的布局关系。
- §3.5.2 的前缀磁盘缓存可复用完整 CSA/HCA compressed entries；最后不完整压缩块的原 token state 仍需重算。SWA 另有 full cache、periodic checkpointing、zero SWA caching 三种空间—重算取舍。

## 固定参考实现和框架性能证据

### DeepSeek 官方参考代码

Pro preview revision 文件：[`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)、[`inference/kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py)。官方文件的可读页面与提交历史显示，`model.py` 的最后更新提交是 2026-04-27 的 `a1fd202`，`kernel.py` 的最后更新是 2026-06-08 的 `5607980`；两者均早于包含配置的 `b5968e9` revision（2026-06-22）。本轮读取代码页面中的实现对象包括 gated `Compressor`、独立 `Indexer`、以 layer id 读取 `compress_ratios` 的 `Attention`、FP8/FP4 GEMM 和 TileLang kernel；它们支持对公开参考实现路径的描述，不支持权重下载、硬件运行或 API 内部行为的结论。

### SGLang v0.5.20 tagged release 与 PR

| 证据 | 固定条件/结果 | 边界 |
| --- | --- | --- |
| [Unified Tree PR #34565](https://github.com/sgl-project/sglang/pull/34565)，收录于 [SGLang v0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20) | DeepSeek-V4-Flash-0731 共享前缀模拟；64 requests / 8 groups，system prompt 24,576 tokens、question 8,192、output 128。开启 out-of-window free 时，token hit rate 从 43.81% 到 60.75%，mean TTFT 1,569.93 ms 到 1,069.58 ms；p95 TTFT 和 input throughput 也由 PR 报告。Release 用四舍五入值 43.8→60.8%、1.57→1.07 s。 | SGLang PR 作者的指定 workload 结果，不是独立复现或任意线上请求分布。 |
| [TRT-LLM DeepSeek-V4 Attention PR #30805](https://github.com/sgl-project/sglang/pull/30805)，收录于同一 release | B200、SM100/103、FP8、TP=1；相对 FlashMLA 的 CSA/HCA attention unit-kernel，release 报告 prefill 约 1.2×、decode 约 1.45×。 | kernel-level backend 对比，不是端到端模型/Agent 加速倍数。 |
| [SM120 DeepSeek-V4 PR #29927](https://github.com/sgl-project/sglang/pull/29927)，收录于同一 release | 4× RTX PRO 6000（SM120）DeepSeek-V4-Flash，batch 1；8K 输入 TPOT 为 36.1 ms→10.5 ms，PR 还报告 32K、128K，以及 TTFT 在 8K–128K 输入下降约 20%。 | PR 改动组合 paged-MQA indexer、FP4 MoE 与 page-split 等路径；TPOT/TTFT 结论绑定具体 backend、输入长度和配置。 |

## 教学与工程解释来源

| 来源 | 阅读用途 | 使用边界 |
| --- | --- | --- |
| Hugging Face, [DeepSeek-V4: a million-token context that agents can actually use](https://huggingface.co/blog/deepseekv4)，2026-04-24，Hybrid attention 一节 | 用层交错、4×/128×压缩和 RoPE/KV 精度解释连接模型整体结构；适合检查面向读者的组织。 | 主要是发布时团队/平台技术解释；关键值与后续状态需回到作者报告和固定代码。 |
| UC Berkeley EECS, [F2026 Lecture 7 slides: Deepseek V4 Configuration](https://scalable-ai.eecs.berkeley.edu/F2026/assets/lecture_slides/lecture7.pdf)，DeepSeek V4 configuration 页 | 将 Pro/Flash 层数、专家数、top-k 与 CSA/HCA/SWA 层计数并排；用于检查比较维度是否对齐。 | 课程 slides 是教学材料，精确结论回到 DeepSeek 报告和模型配置。 |
| NVIDIA Developer Blog, [Build with DeepSeek V4 Using NVIDIA Blackwell and GPU-Accelerated Endpoints](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)，2026-04 | 观察硬件发布和推理配置怎样界定适用条件，帮助区分模型效率估算、kernel/serving 实测与 endpoint。 | 厂商文章只支持其报告的 Blackwell 条件和测试口径，不用于通用性能外推。 |

## 未验证边界

- 报告的 1M benchmark、FLOPs 与 KV cache 数字是作者公开报告值；本轮未下载完整权重或复现模型结果。
- 参考代码是固定公开文件快照；未在本机运行 DeepSeek 推理、TileLang kernel、SGLang PR workload 或硬件 benchmark。
- 未找到可将 CSA、HCA、低精度、MoE/mHC 和 cache layout 的贡献逐一隔离的完整 1M 模型级因果消融，因此正文不把系统级结果归于单一注意力算子。
- SGLang v0.5.20 的三个性能案例来自不同 GPU、baseline、测量层级与负载，不作横向合并。
