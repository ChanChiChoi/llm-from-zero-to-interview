# Qwen3-Omni 30B A3B 来源笔记

> 核验日期：2026-09-22。模型入口来自 Artificial Analysis；本文沿已发现的 `Qwen3 Omni 30B A3B` 条目追踪官方资料，不从官方目录、Hugging Face Trending 或论文列表新增模型。

## 1. 发现入口与证据边界

| 层级 | 证据 | 可确认内容 | 不能直接推出的结论 |
|---|---|---|---|
| 模型发现 | [Artificial Analysis instruct](https://artificialanalysis.ai/models/qwen3-omni-30b-a3b-instruct)、[reasoning](https://artificialanalysis.ai/models/qwen3-omni-30b-a3b-thinking) | 页面存在 `Qwen3 Omni 30B A3B Instruct/Thinking` 配置，以及第三方指数、速度、上下文、价格和模态字段 | AA 字段不是官方参数、训练 recipe、独立硬件 profiling 或裸模型能力 |
| Agent 评测 | [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 当前快照没有精确的 `mini_swe_agent_qwen3_omni_*` 行 | 不能把其他 Qwen、Qwen3.8 或 Qwen3.5 行迁移为 Qwen3-Omni 成绩 |
| 官方模型资料 | [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)、[HF Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 模型家族、组件、安装和推理入口、许可证及变体边界 | README 的运行路径不等于所有生产 kernel 或目标硬件验收 |
| 官方技术资料 | [Qwen 博客](https://qwen.ai/blog?id=65f766fc2dcba7905c1cb69cc4cab90e94126bf4&from=research.latest-advancements-list)、[技术报告](https://arxiv.org/abs/2509.17765) | Thinker-Talker、AuT、TM-RoPE、多码本 Talker、Code2Wav、训练和后训练主线 | 发布方延迟、评测和训练描述不能替代本地复现 |
| API/服务 | [阿里云 Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni) | 托管 API 的输入输出与服务使用边界 | API 契约不等于开放权重实现已在每个后端通过 |

Artificial Analysis instruct 详情页在三条代理上均返回 HTTP 200、`3,833,501` bytes，SHA-256 为 `e62d2c5dac6af3b1b08efa94b58e321df8ac16a813de05d95ad7fabe5eedb316`。对应临时快照为 `/tmp/aa-qwen3-omni-instruct-1234-20260922.html`、`/tmp/aa-qwen3-omni-instruct-7890-20260922.html` 和 `/tmp/aa-qwen3-omni-instruct-8098-20260922.html`。reasoning 详情快照为 `/tmp/aa-qwen3-omni-reasoning-1234-20260922.html`，`3,845,331` bytes，SHA-256 为 `22f186bdd723dda3647aca48c7cbbe2d91af1832f08217620c3dad39bff5c748`。

instruct 页面第三方字段约为 Intelligence Index `6.0061`、输出速度 `94.59 tokens/s`、上下文 `66K`、总参数 `35.3B`、活跃参数 `3B`、输入/输出约 `$0.25/$0.97` 每百万 token，并标为非 reasoning、文本/图像/语音/视频输入和文本/语音输出。这些字段只作为 2026-09-22 的 AA 页面记录，不改写成官方模型事实。

## 2. 模型家族边界

官方资料把同一 Qwen3-Omni 家族拆成不同 artifact，而不是三个互不相关的基础模型：

| 变体 | 公开定位 | 记录规则 |
|---|---|---|
| `Qwen3-Omni-30B-A3B-Instruct` | Thinker + Talker；面向文本、音频、图像、视频理解以及文本/语音输出 | 本章的主要模型入口 |
| `Qwen3-Omni-30B-A3B-Thinking` | Thinker-only，文本输出；用于思考/理解路径 | 作为同家族 Thinker-only 变体，不重复计数为新基础模型 |
| `Qwen3-Omni-30B-A3B-Captioner` | Instruct 下游的音频 caption 微调模型 | 作为下游 artifact，不写成新的 Omni 基础模型 |

官方 HF 配置公开 `35.3B` 量级的总参数字段和约 `3B` active 的 MoE 口径；具体服务页面、参数统计脚本和 provider 的计数口径要分开。面试时应把 total、active、驻留权重、音频/视觉 encoder、Talker 状态和 serving workspace 分账。

## 3. Thinker-Talker：理解与语音生成解耦

Qwen3-Omni 的核心结构是两个协作但职责不同的 MoE 模块：

1. **Thinker** 负责把文本、音频、图像和视频输入组织成理解、推理、工具调用或文本回答。
2. **Talker** 负责把多模态上下文转成流式语音 token，再交给声码器生成 waveform。
3. Talker 不只读取 Thinker 已经生成的高层文本表示，而是使用音频/视觉多模态特征。这样可以让 RAG、function calling、安全过滤器等在 Thinker 输出链路中介入，同时保留更直接的跨模态语音生成路径。
4. 这不是“两个模型简单串联”。它要求调度器同时维护理解侧 token、音频/视觉条件、Talker 的多码本状态、取消/回退和流式 packet 顺序。

面试回答可以写成：

```text
multimodal input
  -> Thinker: understanding / reasoning / tool proposal / text
  -> host: retrieval, function call, policy and verifier
  -> Talker: multimodal-conditioned first codebook + residual codebooks
  -> Code2Wav: causal waveform synthesis
  -> audio packets
```

图中的箭头表示系统数据流，不代表官方已经公开了所有跨模块 tensor 接口或线上调度实现。

## 4. AuT 音频编码器

技术报告将 AuT 作为从零训练的音频编码器，公开了以下主线：

- 训练数据约 2,000 万小时监督音频；报告描述约 80% 中英文伪标注 ASR、10% 其他语言 ASR、10% 音频理解数据。
- 采用 Conv2D 做 8 倍时间下采样。
- 输出音频 token rate 为 12.5 Hz，即约每 80 ms 一个音频时间步。
- 使用动态 attention window，约覆盖 1-8 秒，服务于不同音频片段和流式上下文。
- 编码器约 0.6B 参数，不能与 Thinker/Talker 的 active 参数相加后当作新的单一 active 数字。

这里的面试重点是速率和延迟的关系：12.5 Hz 代表条件序列长度比原始声学帧短很多，但 80 ms 只是 token 时间粒度，不等于端到端语音首包延迟。编码、Thinker prefill、Talker 首码本、Code2Wav、网络 packet 和后端调度仍要分别测量。

## 5. 视觉编码器与 TM-RoPE

视觉 encoder 来自 Qwen3-VL，初始化于 SigLIP2-So400m，报告给出约 543M 参数。音频、视频和图像 token 需要共享一个能表达时间与空间的坐标系统，因此模型使用 TM-RoPE：

1. 三个位置维度分别表示 temporal、height 和 width。
2. rotary angle 的公开分配为 `24/20/20`。
3. 音频的每个 temporal ID 对应约 80 ms。
4. 视频 temporal ID 按真实 timestamp 对齐到 80 ms，而不是只按固定帧序号递增。
5. 这使模型可以处理任意时长的流式输入；报告将它与 Qwen2.5-Omni 的固定 2 秒 chunk 方案区分开。

TM-RoPE 的关键价值不是“加了三个 RoPE”，而是让不同媒体的时间坐标落在同一相对时间轴上。实际部署还要保存媒体时间戳、采样/丢帧策略、chunk 边界、token offset 和 cache/state handle；否则跨 chunk 回放时会产生隐性的时间错位。

## 6. Talker：多码本语音生成与 Code2Wav

语音 codec 使用多码本 residual vector quantization。Qwen3-Omni 的 Talker 采用两级生成：

1. 第一个 codebook 由自回归路径预测，承担主要语音序列决策。
2. MTP 预测其余 residual codebooks，减少完全串行预测的延迟。
3. input/output audio code rate 维持在 12.5 Hz。
4. Code2Wav 使用轻量 causal ConvNet 把音频 code 转成 waveform，替代 block-wise DiT 路径。

因此不能简单说“语音生成就是文本 TTS”。要同时分析首码本的自回归质量、残差码本的并行/预测误差、codec code rate、声码器实时性和 packet 拼接。MTP 降低的是多码本生成的串行深度，不能自动证明所有硬件上的端到端首包或实时率都改善。

## 7. 训练与后训练

报告描述的预训练阶段为：

| 阶段 | 公开主线 | 工程含义 |
|---|---|---|
| S1 | 冻结语言模型，训练 encoder/adapter | 先把不同媒体映射到语言模型可消费的表示空间 |
| S2 | 约 2T tokens 的 text/audio/image/video/video-audio 混合训练 | 建立跨模态理解和跨媒体对齐能力 |
| S3 | 上下文从 8,192 提升到 32,768，并增加长音频/长视频 | 把流式/长媒体场景纳入训练分布 |

Thinker 后训练包括 lightweight SFT、strong-to-weak distillation（报告区分 off-policy/on-policy KL）、GSPO，以及多模态 rule-based/model-based reward。Talker 后训练包括大规模多模态语音数据、continual pretraining + long-context、multilingual DPO 和 speaker fine-tuning。

这些是报告公开的流程类别，不是完整 loss 权重、数据配比、teacher 数量、奖励模型参数或生产训练 recipe。面试中如果被追问，应该明确哪些字段来自报告、哪些仍没有公开。

## 8. 流式 serving 与延迟账本

官方报告给出理论首包延迟：audio first packet `234 ms`、video first packet `547 ms`。这两个数字是论文/发布方理论或实验口径，不是本机实测，也不应直接当成线上 p50/p99。

报告还描述 Thinker/Talker 的异步 chunked prefill，用于提升高并发服务利用率。一个可审计的请求 manifest 至少应保存：

```text
model/revision/variant
media timestamp and sampling policy
Thinker chunk offsets and tool/policy events
Talker codebook state and packet sequence
Code2Wav chunk boundaries and audio format
cancel/retry/replay state
TTFT/first-audio-packet/first-video-packet/TPOT
```

官方 README 当前要求 `transformers>=5.2.0`、`qwen-omni-utils` 和 FlashAttention 2；`disable_talker()` 可节省约 10GB 显存。README 同时说明当前 vLLM serve 主要支持 Thinker，Instruct 音频输出仍处于实现推进阶段。因此“开源仓库可运行”“Thinker 可服务”“Instruct 音频输出已在 vLLM 生产可用”是三个不同状态。

## 9. 面试中的证据分层

| 问题 | 稳妥回答 |
|---|---|
| Qwen3-Omni 是一个模型还是三个模型？ | 一个家族，包含 Instruct、Thinking 和 Captioner 等不同 artifact；不能把下游 captioner 和 Thinker-only 变体都计作新的基础模型。 |
| 为什么需要 Thinker/Talker？ | 理解、工具/策略介入与流式语音生成的约束不同；解耦可以让 Thinker 接入 RAG/function calling/安全过滤，同时让 Talker直接利用多模态特征。 |
| TM-RoPE 解决什么？ | 把音频、视频的真实时间与视觉空间放入统一位置坐标；它解决表示对齐，不等于解决所有长上下文检索或 streaming 调度问题。 |
| MTP 在 Talker 中做什么？ | 首码本走自回归，剩余 residual codebooks 由 MTP 预测，减少码本间的串行深度；质量、接受率和端到端实时率仍需按 codec、后端和硬件实测。 |
| 234 ms/547 ms 能否作为服务 SLO？ | 不能。它们是报告口径的理论首包数字；线上还要加入媒体预处理、排队、网络、取消、重试、工具和 provider 差异。 |
| Qwen3-Omni 有 DataCurve 成绩吗？ | 当前快照没有精确 `mini_swe_agent_qwen3_omni_*` 行，因此不迁移其他 Qwen 行的 Pass@1、成本或 Agent steps。 |

## 10. 待核验项

- 完整 Thinker/Talker 参数账本、专家路由和跨模块 tensor/通信接口。
- AuT、视觉 encoder 和 Talker 的目标硬件 profiling、显存峰值及真实流式吞吐。
- vLLM/SGLang 等后端对 Instruct 音频输出、取消、恢复和多码本状态的完整支持。
- 首码本/残差码本的接受率、Code2Wav 实时率和长音频跨 chunk 误差。
- 完整训练数据配比、loss、teacher/reward 规模、GSPO 细节和 speaker fine-tuning recipe。
- 当前模型 revision、API provider 和固定硬件下的独立多模态/Agent 评测。

## 11. 来源与本地抓取记录

- [Qwen3-Omni GitHub README](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/main/README.md)，本地快照：`/tmp/qwen3-omni-github-readme-20260922.out`。
- [HF Instruct README](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/README.md)，本地快照：`/tmp/qwen3-omni-hf-instruct-readme-20260922.out`。
- [HF Instruct config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json)，本地快照：`/tmp/qwen3-omni-hf-instruct-config-20260922.out`。
- [HF Thinking config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking/raw/main/config.json)，本地快照：`/tmp/qwen3-omni-hf-config-thinking-20260922.out`。
- [技术报告源码](https://export.arxiv.org/e-print/2509.17765)，本地解压目录：`/tmp/qwen3-omni-paper-src/`；PDF 快照：`/tmp/qwen3-omni-technical-report-20260922.out`。

本笔记只把官方报告/模型卡/README 明确披露的字段写成事实；第三方页面、发布方延迟和未下载完整权重/未运行目标后端的项目均保留证据边界。
