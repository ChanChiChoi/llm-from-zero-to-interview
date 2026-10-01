# Qwen3.5-Omni Plus / Flash 来源笔记

> 核验日期：2026-09-24。锚点只取自 Artificial Analysis 已有 `Qwen3.5-Omni-Plus` 与 `Qwen3.5-Omni-Flash` 条目；DataCurve 当前快照未检出精确 Omni 行。以下围绕该锚点阅读 Qwen Team 技术报告与阿里云 Model Studio 文档，不以论文、API 文档或其他仓库新增模型候选。

## 1. 锚点与证据边界

| 层级 | 来源 | 可确认内容 | 不可推出 |
|---|---|---|---|
| 模型发现 | [Artificial Analysis Plus](https://artificialanalysis.ai/models/qwen3-5-omni-plus)、[Flash](https://artificialanalysis.ai/models/qwen3-5-omni-flash) | 榜单中存在 Plus、Flash 两个同家族配置；Plus 页面有 256K context、Index `20.3839890713777`、输出速度 `84.6202668152731 tokens/s`、输入/输出 `$0.40/$4.80` 每百万 token 等当时的第三方/provider 字段 | 不是官方参数、统一硬件吞吐、训练配方或独立 benchmark；不把页面字段差异解释成模型 revision |
| Agent 评测 | [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 2026-09-24 快照未检出精确 `mini_swe_agent_qwen3_5_omni_*` 行 | 不迁移其他 Qwen 的 Pass@1、成本、steps 或 Agent 分数 |
| 技术报告 | [Qwen3.5-Omni Technical Report, arXiv:2604.15804v2](https://arxiv.org/abs/2604.15804v2) | Thinker-Talker、Hybrid MoE、AuT、显式时间戳、ARIA、多阶段训练/后训练及作者评测 | 报告中的数据规模、延迟、基准和效果均是发布方口径，不是本地复现 |
| 服务文档 | [Alibaba Cloud Model Studio: Qwen-Omni](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni) | 托管 API 示例使用 `qwen3.5-omni-plus`；语音生成示例要求 `stream=True`；Plus/Flash 支持 custom voice 的文档边界 | API 字段和限额不披露模型内部架构；未做真实 API 调用 |

本地证据快照：AA Plus `3,826,061` bytes / SHA-256 `b2127d33c029c55eac112636964f2a9cb9196b69ca5f792c90a2a9ae79f387d9`；AA Flash `3,834,828` bytes / `17bd65b3597f29af8a655e00d038b774dc2ae89c821848262b7d0e4bfc8d560c`。DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；最新 Model Studio 页面 `402,771` bytes / `9140a68a7b0a98def6a8cca23e759359895046bd3fc6123b9c361c7526bd7578`；arXiv v2 abstract `42,723` bytes / `7c93a28bd4e5bd795492e20ee6fcbb991920d5834bc6f9f1d4f83dfc0472a34a`。arXiv 源码包 `2,984,159` bytes / `fd53a97d5be7eaa8c981c853f1e6a4faa9b86c69c5fe6f3dbcfa33b66abd81ff`，核读 `content/arch.tex`、`pretraining.tex`、`posttraining.tex` 与 `experiments.tex`。最新网络刷新件位于 `/tmp/*refresh-1234-20260924*`；原始论文源码位于 `/tmp/llm-rankings-rescan-20260924.KVkMMd/`。

## 2. 两代 Omni 的边界

Qwen3.5-Omni 是 Qwen3-Omni 后续家族，不应把前代数字直接沿用：

| 技术面 | Qwen3-Omni（前代报告） | Qwen3.5-Omni（本锚点报告） |
|---|---|---|
| AuT 时间粒度 | 12.5 Hz，约 80 ms | 6.25 Hz，约 160 ms；报告称 4 个 Conv2D 块下采样 16 倍 |
| 音频数据口径 | 前代报告自己的监督音频数据 | AuT 使用 Qwen3-ASR 生成的约 4,000 万小时 audio-text pairs；不能等同于整个 Omni 家族总数据 |
| 时间对齐 | TM-RoPE 时间/空间坐标 | 保留 TM-RoPE，并额外插入秒级文本 timestamp；音频还随机插入 timestamp |
| Talker 文本/语音组织 | 双轨输入设计 | ARIA 把文本和 speech token 组织为一个自适应交错的单序列 |

这里的 6.25 Hz 是音频表示的 token 时间粒度，不是实时率、首包延迟或 ASR 时间误差；不同 encoder 和任务的小时数也不是可以直接相加的互斥数据集统计。

## 3. Thinker-Talker 与 Hybrid MoE

报告延续 Thinker-Talker：Thinker 统一处理文本、音频、图像和视频，生成文本/交互内容；Talker 结合多模态输入和 Thinker 文本生成流式语音。两者都采用 Hybrid MoE Transformer；作者指出其中的 Gated Delta Net（GDN）有助于长音视频序列建模并降低长上下文 KV I/O。完整专家布局、active 参数、路由通信、生产 kernel 和真实吞吐没有由本次来源完整确认。

Thinker 使用 Qwen3.5 tokenizer。报告称这是词表 250K 的 byte-level BPE（此前 150K），并报告多数语言的编码/解码效率提升 10–60%。这是发布方对 tokenizer 的效率主张，不应按同一比例推算每个 prompt 的 token 数，也不是 Omni 专属模块的独立复现。

与普通“文本回答后交给独立 TTS”的系统图相比，Talker 保留来自 Thinker 的多模态上下文，也可以利用当前轮文本流；因此要将 Thinker 语义输出、Talker 多码本状态、声码器状态和播放器 packet 分开记录。论文架构说明不意味着宿主系统已经接入工具、安全门或语音 verifier。

## 4. AuT：降低序列速率，保留在线与离线两种工作点

报告描述的 AuT 从零训练，是 attention encoder-decoder 音频 Transformer：

- 输入由 4 个 Conv2D 块下采样 16 倍，输出约 `6.25 Hz`，即约每 `160 ms` 一个时间步。
- 训练使用约 4,000 万小时、由 Qwen3-ASR 生成的 audio-text pairs；报告称数据覆盖 20 多种语言，中文、英文、多语言数据比例约 `3.5:3.5:3`。
- 动态 attention window 用于平衡实时 prefill cache 与离线音频理解。

低 token rate 可以缩短多模态序列，但不单独保证低端到端延迟：编码器 chunk、Thinker prefill、队列、Talker 首个可播放 chunk、codec 解码、传输和客户端缓冲都要测。

## 5. TM-RoPE 加显式 timestamp：把稀疏位置 ID 变成可读时间锚点

Qwen3.5-Omni 保留 temporal/height/width 位置坐标，但报告指出长视频按绝对时间映射 temporal position ID 时，视觉 patch 的 ID 会很稀疏，削弱长程建模；为不同帧率做均匀大规模采样也增加数据构建成本。其补充做法是：

1. 在每个视频或音视频时间 patch 前加入秒数格式的文本 timestamp；
2. 在音频序列中按随机间隔插入 timestamp，帮助跨模态时间对齐；
3. 音频 temporal ID 按约 160 ms 递进，视频 temporal ID 依据实际 timestamp 动态对齐到同一 160 ms 分辨率；
4. 不同模态的 position 编号连续推进，空间 height/width ID 仍用于视觉 patch。

这是“位置编码 + 显式文本时间码”的组合，不是抛弃 TM-RoPE。工程评测应把 fps、动态采样、timestamp 精度、音画偏移、丢帧与跨 chunk replay 一并固定；只加时间字符串不能证明模型会可靠定位任意视频事件。

## 6. Talker 与 ARIA：按样本级 token 比率约束流式对齐

Talker 以 RVQ speech code 为输出基础，MTP 预测当前帧的 residual codebooks，之后由因果、可流式 ConvNet codec 解码 waveform。专用 Talker system prompt 指定目标声音属性，报告称可支持 zero-shot voice cloning 与可控语音；这些是报告方法/能力声明，不等同于第三方语音质量复现。

**ARIA（Adaptive Rate Interleave Alignment）** 是本报告最值得面试展开的新方法：它将文本和语音从双通道改为单个交错序列。对生成序列的任意前缀，累计 speech:text token 比率不得超过该样本级的全局比率。它不用 MFA 离线对齐，也不规定固定 interleave rate，而是适应不同语言中 tokenizer 效率不一致的情况；目标是减少流式语音中的漏词、发音错误、数字表达歧义和双轨同步开销。该约束来自作者报告；公开资料没有提供可独立复现的完整训练 loss、约束实现或跨 tokenizer 消融。

面试时要区分三件事：ARIA 约束的是文本/语音 token 的生成节奏；MTP 降低 residual codebook 的串行预测深度；Code2Wav 负责把 codec token 转为 waveform。三者优化对象不同，不能笼统称作一个“低延迟语音算法”。

## 7. 训练与后训练账本

### 预训练

| 阶段 | 报告描述 | 解读边界 |
|---|---|---|
| S1 Encoder Alignment | 冻结 LLM，分别训练视觉/音频 encoder 与 adapter | 对齐媒体表示，不是完整多模态训练 |
| S2 General | 约 4T tokens；列出的 text `0.92T`、audio `1.99T`、image `0.95T`、video `0.14T`、video-audio `0.29T` | 列项算术和约 `4.29T`，原文仍称约 4T；保留原口径，不猜测舍入/重叠规则 |
| S3 Long Context | 最大序列长度从 `32,768` 提升至 `262,144`，增加长音频/长视频比例 | 是报告描述的阶段配置，不是 API 一次请求的通用保证 |

报告摘要另称使用超过 1 亿小时音视频内容；AuT 的 4,000 万小时 audio-text pairs 和 Talker 的超过 2,000 万小时多语言语音数据是不同训练环节/口径。它们可能有子集或数据处理重叠，不能简单相加为独立原始时长。

### 后训练

- Thinker：专家教师分别 SFT/RL 后做 specialist distillation；再用文本条件回答作为相应音频输入 query 的 on-policy distillation target；最后用多轮 interaction-aligned RL 处理语言切换、persona 一致性和长上下文指令遵循。
- Talker：超过 2,000 万小时多语言语音与多模态上下文训练；高质量长上下文 CPT，并借助 Qwen3-Omni-Captioner 缓解噪声数据造成的幻觉；之后 multilingual DPO、rule rewards + GSPO、speaker fine-tuning。
- 公开的是阶段与目标类别，而不是数据去重、loss 权重、teacher/reward 规模或完整生产 recipe。

## 8. 评测、首包与 API 合同

论文摘要报告 Qwen3.5-Omni-Plus 在 215 项音频/音视频理解、推理和交互子任务及 benchmark 上的结果，并声称支持超过 10 小时音频理解、400 秒 720p/1 FPS 视频。论文表格的首包数是在内部 vLLM、启用 `torch.compile` 与 CUDA Graph（MTP/codec 路径）的报告口径：并发 1 时 Flash audio/video `235/426 ms`，Plus `435/651 ms`。Plus 与 Flash 使用不同部署资源和并行策略，作者明确不建议严格横向比较；这些也不是本机硬件 profile 或线上 SLO。

Model Studio 当前文档将 Qwen3.5-Omni 定位为多模态理解与语音输出服务：示例 API ID 为 `qwen3.5-omni-plus`，生成语音示例要求 `stream=True`，文档只对 Plus/Flash 声明支持 custom voice 且 snapshot version 不支持。服务限额按输入方式/任务分列，不能把论文支持时长直接当成 API request limit。此处没有真实 endpoint probe，也没有验证返回的模型 revision、音质、首包或实时率。

## 9. 面试复述框架

```text
multimodal input
  -> Thinker (Hybrid MoE): understand / reason / text stream
  -> Talker (Hybrid MoE): multimodal-conditioned RVQ speech tokens
       -> ARIA: one adaptive text-speech interleaved stream
       -> MTP: residual codebooks for current frame
       -> causal streaming codec: waveform chunks
  -> transport/player: packet, buffering, cancellation and playback checks
```

推荐面试回答顺序：先讲 ARIA 解决双轨 token 速率失配，再讲 160 ms AuT 时间粒度和“TM-RoPE + 明确秒级时间码”，最后讲 MTP/codec 与端到端延迟账本。把数据规模、215 项 benchmark 和首包延迟都标为发布方报告值。

## 10. 尚未闭环的实现问题

- Plus/Flash 的精确层数、专家配置、总/active 参数、KV/state 布局和通信策略。
- ARIA 约束的训练目标、语言级 token ratio 分布、与 MFA/fixed-rate 的独立消融及用户研究。
- AuT 分块缓存的 eviction、时间戳抖动/丢帧误差、10 小时音频真实检索质量。
- Talker residual MTP acceptance、codec chunk seam、不同硬件实时因子和声学质量。
- 完整数据配方、后训练奖励、模型/数据版本、公开权重及固定 artifact runtime。
- 目标硬件上的首包/TPOT、API response model、并发成本、取消/恢复和独立复现。

状态：**AA 单榜内容专题闭环 + 官方技术报告/API 文档核验**；不代表 DataCurve Agent 闭环、完整架构/训练 recipe、权重加载、独立评测或生产验收。
