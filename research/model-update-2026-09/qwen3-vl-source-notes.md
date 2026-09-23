# Qwen3-VL-235B-A22B：视觉语言、长视频位置与多模态 Agent 证据摘记

核验日期：2026-09-22。本文只记录已经由 Artificial Analysis 发现的 `Qwen3-VL-235B-A22B` 及其官方周边资料。DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_qwen3_vl_*` 行，因此不迁移其他 Qwen 模型的 Agent 分数。

## 1. 候选身份和证据边界

| 层级 | 已核验事实 |
|---|---|
| 发现入口 | [Artificial Analysis instruct](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-instruct) 与 [reasoning](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-reasoning) |
| 榜单日期 | AA 页面显示 `releaseDate=2025-09-23`；这是第三方目录字段，不代替 Qwen 官方发布日期 |
| AA instruct | Intelligence Index `9.93699415502801`，约 `50.9368422526377 tokens/s`，`262,144` context，`235B total / 22B active`，页面价格约 `$0.40/$1.60` 每百万 input/output token |
| AA reasoning | Intelligence Index `13.4375683732418`，约 `54.6738111727239 tokens/s`，`262,144` context，`235B total / 22B active`，页面价格约 `$0.40/$4.00` 每百万 input/output token |
| DataCurve | 当前快照没有精确 `mini_swe_agent_qwen3_vl_*` 行；不迁移其他 Qwen 的 Pass@1、成本或 Agent steps |
| 模型归并 | Instruct 与 Thinking/Reasoning 是同一 Qwen3-VL 基础模型的不同运行配置或 artifact，不计为两个基础模型 |
| 当前状态 | **AA 单榜内容专题闭环**；研究笔记、官方技术报告、模型卡/配置、正式章节和书系配套均已具备，但完整权重加载、生产 kernel、目标硬件 profiling、端到端多模态 serving、GUI/tool acceptance 和独立复现仍待核验 |

AA 中文首页三条代理取得逐字节一致快照：`1,798,627` bytes，SHA-256 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`。instruct 详情为 `3,832,139` bytes，SHA-256 `a58d3922d8e872c76684314e9de21eee78e431ee0d527d10c65cadc9f7755efa`；reasoning 详情为 `3,849,165` bytes，SHA-256 `28f97633dfbeb145e696b9f87a01ba891755d80bfaaf7e7a1db4d8d7ecaa59b3`。这些哈希用于固定本轮第三方目录观察，不把指数或速度写成模型固有常数。

两个排行榜的证据要分账：AA Intelligence Index、速度、价格和 context 是第三方页面/provider 字段；DataCurve 的 `mini-swe-agent` 结果若未来出现，也会绑定 effort、工具、任务环境、仓库和 verifier。不能把它们合并成一个裸模型分数，也不能把其他 Qwen 的 Agent 行迁移到 Qwen3-VL。

## 2. 官方 artifact、revision 和配置

官方仓库和模型卡入口如下：

1. [Qwen3-VL 官方 GitHub](https://github.com/QwenLM/Qwen3-VL)：README 固定技术报告、官方博客、模型卡、Cookbook 和部署入口。
2. [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)：revision `710c13861be6c466e66de3f484069440b8f31389`，`lastModified=2025-11-26T13:18:18Z`。
3. [Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)：revision `6664affde68449468deb7527186455c7450c13c0`，`lastModified=2025-11-26T13:18:17Z`。
4. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)：arXiv `2511.21631`，当前阅读版本 v2，2025-11-27 修订。
5. [Qwen3-VL 官方博客](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list)：本轮页面返回通用站点壳，正文事实以论文、GitHub README 和 HF 模型卡为主，不把壳页面当作独立证据。

固定文件哈希：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| Instruct README | 6,697 bytes | `207f81bc285522284cb14c0b8ff2ff022993f5e55c39edf570eb00e1c64103d1` |
| Thinking README | 6,697 bytes | `19653aec7f696a10b27868d210a829c4825dd766121c76c524173096684dbe1` |
| Instruct/Thinking config | 1,663 bytes | `a9b0434bf1bb5de21786cb3dc3abf95e7254a1e131557c911d7271d2fb63e476` |
| Technical Report abstract HTML | 52,215 bytes | `6305249961155b67c55807b74b78e2cdfee506ef25b146eb48c5f7e2a90d3640` |
| Technical Report PDF | 4,385,558 bytes | `ee075d08e67de1148d6437c6c1d481f7894183b8793905a2deb5f62664f49380` |
| arXiv HTML v2 | 613,623 bytes | `a22c5f2d9070182dd6758369f423feb6953dbe1855baf391ae88ff3a909cff48` |
| arXiv API record | 6,749 bytes | `670f537e45bb9d4d5d96f4b85b62b92ad4d960f829f3b49f49272a0595514fd5` |

当前 HF config 的关键字段为：

| 子系统 | 公开配置 |
|---|---|
| 模型类 | `Qwen3VLMoeForConditionalGeneration` |
| 文本 decoder | 94 layers，hidden size 4096，64 query heads / 4 KV heads，head dim 128 |
| 文本 MoE | 128 experts，top-8 routing，MoE intermediate size 1536 |
| 位置与精度 | `max_position_embeddings=262144`，RoPE theta `5,000,000`，BF16 |
| M-RoPE | `mrope_section=[24,20,20]` |
| 视觉 encoder | depth 27，hidden size 1152，patch size 16，temporal patch size 2 |
| 视觉合并 | spatial merge size 2，output hidden size 4096 |
| DeepStack | 中间视觉层索引 `[8,16,24]` |

配置可以支持结构字段、上下文上限和视觉 patch 形状；它不能单独证明训练数据规模、完整参数账本、生产 kernel、专家负载均衡或目标硬件吞吐。AA 的 `262,144` context、config 的最大位置和论文 S3 的 262K adaptation 也要绑定各自来源，不能写成“已经验证 1M 原生上下文”。

## 3. 三模块结构：视觉、合并器和语言模型

Qwen3-VL 的基本计算图是：

```text
image / video
    -> SigLIP2 vision encoder
    -> two-layer MLP vision-language merger
    -> visual tokens + DeepStack residual features
    -> Qwen3 text decoder / MoE routing
    -> text or thinking/tool response
```

视觉编码器把图像或视频 patch 转成视觉表示；两层 MLP merger 将视觉 hidden size 投影到语言模型 hidden size；Qwen3 LLM 再对文本 token、视觉 token 和时间/空间位置进行统一建模。三模块拆分让视觉预处理、视觉 token 数量和语言 decoder 的上下文预算可以分别测量。

## 4. Interleaved-MRoPE：改变频率分配，而不是删掉空间轴

多模态模型要同时表达时间、图像高度和图像宽度。普通文本 RoPE 只有一条序列位置；早期 MRoPE 可以把 temporal、height、width 对应的频率维度分成连续块。Qwen3-VL 的 Interleaved-MRoPE 改为在低频和高频维度中交错分配 temporal/height/width 轴，让不同轴在频谱上更均衡。

它解决的不是“视频 token 太多”这个单一问题，而是长视频中位置频率分工可能产生的偏置：如果某一轴集中占据一段频率，模型可能更容易依赖某类局部或短期位置模式。交错分配让时间和空间信息在不同频率范围都有表示。面试时应说清楚：它改变的是 rotary frequency allocation；并没有把视频理解变成无位置的集合，也没有自动解决视频检索或长上下文容量。

配置中的 `mrope_section=[24,20,20]` 是实现字段，不能只凭数字推导论文全部的频率排列；论文正文和上游代码共同支持“temporal/height/width 三轴 + interleaved 频率”的结论。

## 5. DeepStack：把中间视觉特征送进早期语言层

只把视觉 encoder 最后一层经过 merger 后放到语言序列前面，会让视觉信息在进入 LLM 时已经被压成单一路径。DeepStack 从视觉 Transformer 的三个中间层取特征，分别经过专用 merger，再以残差方式注入 LLM 的前几层 hidden states。当前配置的索引是 `[8,16,24]`，上游 Transformers 实现确认这些视觉 embedding 在语言模型前几层注入。

可以把每个早期语言层抽象为：

```text
h_l = Transformer_l(h_{l-1} + DeepStack(vision_layer_k))
```

真实实现包含层映射、投影和 token 对齐，公式只是解释数据流。DeepStack 的关键收益是保留多层视觉抽象：低层可能保留纹理/局部几何，中层包含区域关系，高层更接近语义。它不额外增加视觉 token 的序列长度，因此主要新增的是早期层的 residual 分支、投影和显存/算力，而不是把同一图像复制成三份上下文 token。

## 6. Video Timestamp：让时间信息成为可读的输入证据

对长视频仅依赖 position id，模型看到的是稀疏 token 序列，不一定能直接恢复真实秒数。Qwen3-VL 在视频时间 patch 前加入文本时间戳，例如 `<3.0 seconds>`；训练中同时使用 seconds 和 HMS 格式，让模型学习“该视觉片段发生在什么时候”。

它与 Interleaved-MRoPE 是互补关系：

| 机制 | 解决的问题 | 不负责的事情 |
|---|---|---|
| Interleaved-MRoPE | temporal/height/width 位置频率的表示分配 | 不负责读取外部视频、修复丢帧或决定采样率 |
| Video Timestamp | 将可解释的绝对时间锚点写入视频 token 上下文 | 不负责自动扩大 context、生成字幕或保证时间定位正确 |

生产 serving 仍要保存视频 revision、采样率、frame index、真实 timestamp、chunk offset、重复/丢失帧和回放状态。把 `<3.0 seconds>` 写入 prompt 不等于宿主已经完成时间轴审计。

## 7. 训练 curriculum 和损失账本

技术报告给出四个预训练阶段：

| 阶段 | 目标 | 公开规模/长度 |
|---|---|---|
| S0 | 只训练 merger，使视觉与语言表示接上 | 约 67B tokens，8K context |
| S1 | 全参数 multimodal pretraining | 约 1T tokens，8K |
| S2 | 全参数 long-context pretraining | 约 1T tokens，32K |
| S3 | ultra-long adaptation | 约 100B tokens，262,144 context |

报告还描述 square-root normalized per-token loss，用来平衡 text-only 与 multimodal 数据在 token 规模差异下的贡献。它表达的是数据源归一化策略，不等于一个完整公开的优化器、采样器或 batch 配方。

后训练从 32K 扩展到 256K，并结合 text-only strong-to-weak distillation、Reasoning RL 与 General RL。Reasoning RL 使用 SAPO；General RL 使用规则 reward 和 model-based reward。公开资料支持这些阶段和目标，但没有公开足以独立复现的全部奖励权重、采样温度、rollout 数量、teacher 配置和分布式细节。

## 8. Thinking with Images：工具调用 reward 是必要约束

Qwen3-VL 的视觉推理训练不是简单地把图片问答数据加入 SFT。报告描述约 10K grounding cold-start examples，使用 Qwen2.5-VL-32B 做 visual-agent SFT 与 tool-integrated RL，再蒸馏约 120K multi-turn agent interactions。相关数据包括 GUI 多步轨迹、multimodal function-calling、image/text search、grounding、OCR、文档解析、代码和视觉 STEM。

视觉 Agent 的 reward 至少分三类：

1. answer accuracy：最终回答是否正确；
2. multi-turn reasoning：多轮计划、证据读取和推理是否合理；
3. tool-calling reward：工具是否在正确时机、以合理次数和参数被调用。

报告明确指出，若只有前两类 reward，模型可能学会固定只调用一次工具；tool-call reward 用来约束探索次数与任务复杂度匹配。这个例子很适合面试：最终答案正确并不代表轨迹高质量，工具调用次数也不能孤立地越多越好。

## 9. Agent 与工程边界

官方 README 提供 GUI、手机 Agent、OCR、文档解析、视频理解、2D/3D grounding、Thinking with Images 和 multimodal coding 等 Cookbook 方向。它们说明模型适合成为视觉 Agent 的推理核心，但“能生成工具调用”仍不等于“工具已经执行”。完整链路应写成：

```text
visual input -> model proposal -> schema/permission gate
-> GUI/search/code executor -> observation replay
-> reasoning/tool-call verifier -> final artifact
```

模型负责提出计划和参数；宿主负责权限、浏览器/GUI 操作、网络、文件沙箱、超时、幂等、审计和业务验证。尤其是 GUI Agent，要额外记录截图 revision、坐标缩放、窗口状态、动作序号和副作用确认。论文中 agent interaction 的数量是训练数据披露，不是目标环境中的成功率或独立 Agent benchmark。

## 10. 上游实现证据和部署边界

本轮抓取 Transformers 上游 raw main 作为 2026-09-22 代码快照：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `configuration_qwen3_vl.py` | 6,169 bytes | `200b5e015b9e4b6e425122ac0359e3c6281e4a7638a00d8ebfbe853d61dc1bc7` |
| `modeling_qwen3_vl.py` | 71,312 bytes | `91e5864b5729d52f766b27f1c4cc37efcf8376822935534ebfc8e3b87bf47fc3` |
| `processing_qwen3_vl.py` | 9,573 bytes | `02d50224d9dc9ce38690cfa6a0106be897f619444ebe3ff7ba38cbcdda9e3ef2` |

代码核对到三点：`get_rope_index` 生成 temporal/height/width position ids；视频 grid 按 timestamp 拆分；processor 将 frame timestamp 写进文本 placeholder；DeepStack embedding 在语言模型前几层注入。因为本轮没有固定 upstream commit SHA，这些哈希应标为 raw main 快照，不能伪装成不可变版本。

部署验收要分层：

```text
model card/config -> tokenizer/processor -> full-weight load
-> visual preprocessor -> text/vision numerical correctness
-> DeepStack/M-RoPE/video timestamp replay -> target backend profile
-> GUI/tool acceptance -> artifact/verifier acceptance
```

“HF 模型卡可见”不能证明完整权重已加载；“Transformers 代码支持”不能证明 vLLM/SGLang 的生产 kernel、视频长上下文或 GUI tool harness 已通过；论文 benchmark 也不能替代目标 GPU 的显存、TTFT、TPOT、p99 和失败恢复测量。

## 11. 仍待核验

- 完整 235B/22B artifact 的权重加载、dtype、sharding 和 target hardware memory profile。
- Interleaved-MRoPE 在不同视频长度、采样率和跨 chunk replay 下的消融曲线。
- DeepStack 投影的真实 kernel、激活显存和不同视觉 token 数量下的收益。
- Video Timestamp 对时间定位、长视频召回、重复/丢帧和 OCR 的独立对照。
- SAPO、General RL reward、tool-call reward 的完整超参和可复现训练实现。
- GUI/mobile Agent 的宿主动作执行、权限、坐标映射、重试、幂等和独立 verifier acceptance。
- vLLM/SGLang/Transformers 各后端的完整多模态流式 serving、TP/EP、KV cache 和吞吐/p99。

## 12. 书系映射

- 正式专题：第二十一册第 92 章 [`Qwen3-VL：Interleaved-MRoPE、DeepStack 与视频时间戳`](../../book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)。
- 多模态基础：第四册和第十五册的视觉 token、视频时间轴、processor 与模态预算。
- Reasoning：第十六册的 SAPO、视觉推理与 reward 分账。
- Agent：第十七册和第二十册的 GUI/tool executor、轨迹、权限、回放和 verifier。
- Serving：第六册和第二十四册的 multimodal prefill、DeepStack residual、KV/cache、TP/EP 与目标硬件验收。

## 13. 来源清单

1. [Artificial Analysis Qwen3-VL instruct](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-instruct)
2. [Artificial Analysis Qwen3-VL reasoning](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-reasoning)
3. [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
4. [Qwen3-VL Technical Report, arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
5. [Qwen3-VL-235B-A22B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
6. [Qwen3-VL-235B-A22B-Thinking model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)
7. [Transformers Qwen3-VL configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py)
8. [Transformers Qwen3-VL modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
9. [Transformers Qwen3-VL processing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)

Qwen 官方博客页面本轮只返回通用站点壳，因此正文中的技术结论优先绑定技术报告、固定 HF config/model card、官方 GitHub README 和上游实现快照；不能把博客壳页或第三方目录字段扩写成未公开的参数/训练事实。
