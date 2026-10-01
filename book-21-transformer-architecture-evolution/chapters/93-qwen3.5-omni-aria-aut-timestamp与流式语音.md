# 第 93 章 Qwen3.5-Omni：ARIA、AuT 时间粒度与流式语音

> 核验日期：2026-09-24。锚点来自 Artificial Analysis 已有的 Qwen3.5-Omni-Plus/Flash 条目；DataCurve 当前快照没有精确 Omni Agent 行。核心技术证据来自 Qwen Team 的 [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804v2)，服务合同来自 [Alibaba Cloud Model Studio](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni)。论文中的性能、训练规模和 benchmark 均按发布方口径描述。

与前代 [Qwen3-Omni（第 91 章）](91-qwen3-omni-thinker-talker-aut与流式多模态.md) 分开：本代 AuT 是 6.25 Hz/约 160 ms，前代报告的 12.5 Hz/约 80 ms 和首包数字不能移植到本代。

## 93.1 先把模型身份和接口口径分开

Artificial Analysis 有 Plus、Flash 两个榜单配置。本章把它们视作同一 Omni 家族的服务/规模变体，而不是因两个 slug 就计成两个独立基础模型。Plus 页面当时列出 256K context、Intelligence Index `20.3839890713777`、输出速度 `84.6202668152731 tokens/s` 和 `$0.40/$4.80` input/output 等第三方/provider 字段；这些不是 Qwen 官方参数表，也不是统一硬件的延迟对比。DataCurve 没有精确 `mini_swe_agent_qwen3_5_omni_*` 行，不能迁移其他 Qwen 的 Agent 成绩。

Model Studio 当前语音生成示例使用托管 API ID `qwen3.5-omni-plus`，且要求 `stream=True`；文档称 Plus/Flash 支持 custom voice，但 snapshot version 不支持。托管 API 名称和榜单 slug 属不同命名空间；服务示例不能证明相同权重、内部模块或本地 runtime 可用。

## 93.2 Thinker-Talker：理解与语音生成协同但分工不同

Qwen3.5-Omni 延续 Thinker-Talker，但 Thinker 和 Talker 都采用 Hybrid MoE Transformer：

```text
text / image / audio / video
             |
             v
 Thinker (Hybrid MoE): multimodal understanding + text stream
             |
   multimodal context + current-turn text
             |
             v
 Talker (Hybrid MoE): RVQ speech tokens
       -> MTP residual codebooks
       -> causal streaming codec
       -> waveform packets
```

Thinker 不只是把语音转成文本后交给一个无状态 TTS：报告描述的 Talker 会读多模态上下文和当前轮文本流，生成与对话语境相关的语音。系统要分开维护 Thinker 的上下文/文本、Talker 多码本状态、声码器 chunk 与播放器 packet；论文结构图不等于已经验证了工具、安全或 verifier 接口。

Thinker 的文本输入使用 Qwen3.5 tokenizer：报告称其为 byte-level BPE、词表由 150K 扩为 250K，并在多数语言上带来 10–60% 的编码/解码效率提升。这是论文引用的 tokenizer 级主张，不意味着所有任务的上下文 token 都按相同比例减少，也不是 Omni 独有模块的独立复现。

报告还称 Gated Delta Net（GDN）有助于长音视频序列建模并降低 KV I/O。没有完整层布局与目标硬件实测时，不要据此断言具体 KV 节省比例、吞吐或并发上限。

## 93.3 AuT：6.25 Hz 不是“160 ms 出声”

Qwen3.5-Omni 的 Audio Transformer（AuT）从零训练，使用 4 个 Conv2D 块将输入特征下采样 16 倍，输出频率约 `6.25 Hz`：

```text
1 / 6.25 seconds = 0.16 seconds per encoded time step
```

报告称 AuT 使用约 4,000 万小时、由 Qwen3-ASR 生成的 audio-text pairs，覆盖 20 多种语言；动态 attention window 同时考虑实时 prefill cache 与离线理解。160 ms 是编码后时间粒度，不是端到端语音响应时间。真实首包路径还包括音频预处理、encoder chunk、排队、Thinker prefill、Talker 首个可播放 chunk、codec 解码、传输和播放缓冲。

## 93.4 显式 timestamp 与 TM-RoPE：表示坐标之外再给模型时间码

模型继续保留 temporal/height/width 的 TM-RoPE。报告指出，长视频以绝对时间构造位置 ID 会令 patch temporal ID 变得稀疏；按 fps 均匀构造海量训练样本又有数据成本。Qwen3.5-Omni 因此在视频/音视频 patch 前加入秒数格式的文本时间戳，并在音频序列中随机插入 timestamp。

音频时间 ID 约每 160 ms 一格；视频依据实际 timestamp 动态映射到相同分辨率。于是模型同时拥有：

- RoPE 的相对时序/空间结构；
- 供跨模态对齐的统一时间粒度；
- 可被语言模型读取的显式秒数锚点。

面试时可以说它降低了长视频绝对 position ID 过稀对建模的压力，但不能说它“保证了视频检索”。动态抽帧、VFR、音画偏移、缺帧、重复帧和 chunk replay 仍要作为输入契约评测。

## 93.5 ARIA：让文本和 speech token 共享一个自适应序列

**ARIA = Adaptive Rate Interleave Alignment。** 传统双通道生成分别推进 text 和 speech，文本 tokenizer 与 speech tokenizer 的 tokenization rate 不均衡时容易发生不同步。固定交错频率也难适应不同语言。ARIA 将二者并入单条 interleaved stream，并对生成前缀施加约束：

> 任一生成前缀中累计的 speech:text token 比率，不得超过该样本级的全局比率。

这不是固定的“每 N 个文字 token 插入一个语音 token”，也不是 MFA 离线强制对齐；它是依据样本级速率上限约束逐步生成。作者将目标关联到跨语言对齐、减少漏词/错读/数字歧义和流式同步开销。公开报告没有给出足以独立复现的完整 loss、各语言 ratio 分布或全部消融，因此面试中把原理与效果声明分开。

还要分清两个层次：ARIA 解决 text/speech 两类 token 的相对调度；MTP 预测当前 RVQ 帧的 residual codebooks；因果 codec 再把 codes 解码为 waveform。不能把 MTP 的并行化直接当成用户端首包收益。

## 93.6 三阶段训练与多轮交互后训练

预训练流程可概括为：

```text
S1: freeze LLM; align audio / vision encoders and adapters
  -> S2: ~4T mixed multimodal tokens, sequence length 32,768
  -> S3: sequence length 262,144 + more long audio/video
```

报告列出的 S2 模态数字是 text `0.92T`、audio `1.99T`、image `0.95T`、video `0.14T`、video-audio `0.29T`，简单相加约 `4.29T`；原文称约 4T。应照录为报告的近似/分类口径，不猜各项是否重叠或采用了不同去重方法。

后训练有两条任务主线：

1. **Thinker**：专家教师蒸馏；对音频 query 做 on-policy distillation，以同一 paired query 的文本条件高质量回答作为 target；再以多轮 interaction-aligned RL 改善语言切换、persona 稳定性和长对话指令遵循。
2. **Talker**：多语言语音与多模态上下文训练、高质量长上下文 CPT、DPO、规则奖励 + GSPO、speaker fine-tuning。

论文摘要提到超过 1 亿小时音视频数据；AuT 的 4,000 万小时和 Talker 的 2,000 万小时以上是不同训练环节/数据口径，不能相加成新的总时长。完整数据清洗、去重、loss 权重和 reward 配置仍未公开。

## 93.7 报告延迟、托管 API 和生产验收不能混为一谈

作者称 Plus 在 215 项音频/音视频任务和子任务上取得结果，并报告超过 10 小时音频理解、400 秒 720p/1 FPS 视频。内部 vLLM + `torch.compile` / CUDA Graph 的首包表在并发 1 时为：

| 变体 | audio input | video input | 来源边界 |
|---|---:|---:|---|
| Flash | 235 ms | 426 ms | 报告中的作者部署/理论测量 |
| Plus | 435 ms | 651 ms | 资源配置与 Flash 不同，非严格横向对照 |

它们不是线上 p50/p99、也不是本机结果。端到端至少还要记录媒体上传与解码、queue、首个文本 token、首个可播放音频 chunk、codec、网络、播放器缓冲、并发数和失败恢复。

Model Studio 文档示例能确认托管接口要求 `stream=True`，不能确认隐藏权重、内部 ARIA 状态或目标硬件。论文里的输入时长能力也不能直接代替 API 按请求类型定义的文件和时长限额。没有真实 API key / endpoint probe 时，response model、音质、latency 和恢复行为都保持 `unverified`。

## 93.8 面试题与常见误区

| 问题 | 回答抓手 |
|---|---|
| ARIA 解决什么？ | 语言间 text/speech token rate 不匹配；把两通道合成单流，用样本级全局 ratio 给任意前缀设上限。 |
| 为什么 TM-RoPE 外还要文本 timestamp？ | 缓解长视频绝对 temporal ID 稀疏，给音视频对齐显式、可读的秒级锚点；不保证检索准确。 |
| 6.25 Hz 等于 160 ms 延迟吗？ | 否，只是 encoder token 的时间粒度；首包还要走 Thinker、Talker、codec、传输和播放器。 |
| 4,000 万、2,000 万、1 亿小时能否相加？ | 不能；它们描述 AuT pairs、Talker 多语言语音和报告总体音视频数据的不同阶段/口径，可能有重叠。 |
| MTP 是 ARIA 吗？ | 不是。ARIA 调度 text/speech token；MTP 预测同一 RVQ frame 的 residual codebooks；codec 负责 waveform。 |
| Plus/Flash 首包差异说明 Flash 更快吗？ | 不能仅凭论文表下此结论；作者说明资源配置和并行策略不同，且不是独立同硬件对照。 |

更完整的 source ledger、数据口径和未验证项见 [`qwen3.5-omni-source-notes.md`](../../research/model-update-2026-09/qwen3.5-omni-source-notes.md)。
