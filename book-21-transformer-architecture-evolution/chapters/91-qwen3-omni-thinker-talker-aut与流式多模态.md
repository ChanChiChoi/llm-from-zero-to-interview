# 第 91 章 Qwen3-Omni：Thinker-Talker、AuT 与流式多模态

> 核验日期：2026-09-22。模型发现入口是 [Artificial Analysis 的 Qwen3 Omni 30B A3B 条目](https://artificialanalysis.ai/models/qwen3-omni-30b-a3b-instruct)。DataCurve 当前没有精确的 `mini_swe_agent_qwen3_omni_*` 行，因此本章不写其他 Qwen 配置的 DeepSWE 成绩。

## 91.1 先拆开模型家族和运行配置

Qwen3-Omni 最容易被误解成“一个支持语音的普通多模态 LLM”。公开资料显示，它把理解、推理和语音生成组织成 Thinker-Talker 协作结构，并且存在 Instruct、Thinking、Captioner 等不同 artifact：

| 变体 | 主要职责 | 本章的处理 |
|---|---|---|
| `Qwen3-Omni-30B-A3B-Instruct` | Thinker + Talker；多模态输入与文本/语音输出 | 主要研究对象 |
| `Qwen3-Omni-30B-A3B-Thinking` | Thinker-only，文本输出 | 同家族变体，不新增基础模型 |
| `Qwen3-Omni-30B-A3B-Captioner` | 音频 caption 下游微调 | 下游 artifact，不新增基础模型 |

Artificial Analysis instruct 页面还给出约 `35.3B` total、`3B` active、`66K` context、约 `94.59 tokens/s` 等字段。这些是第三方页面/provider 配置字段，不能直接替代官方参数统计、端到端吞吐或线上 SLO。

## 91.2 为什么是 Thinker-Talker，而不是文本 TTS 串联

最小系统图如下：

```text
text / image / audio / video
          |
          v
   Thinker: understand, reason, call tools
          |
   RAG / function calling / safety / verifier
          |
          v
   Talker: multimodal-conditioned audio codes
          |
          v
   Code2Wav: causal waveform synthesis
          |
          v
       audio packets
```

Thinker 负责把不同媒体转成可推理的语义和行动计划；Talker 负责低延迟、连续的语音 token 生成。关键点是 Talker 不只消费 Thinker 的高层文本表示，还使用音频/视觉多模态特征。这种设计给宿主系统留下介入位置：RAG 可以补充证据，function calling 可以执行外部动作，安全过滤器可以阻止或改写行动，然后 Talker 再按允许的上下文流式生成声音。

这带来一个面试上的区分：

```text
模型输出了“要调用工具” != 工具已经执行
Thinker 生成了文本       != Talker 已经完成音频包
音频 code 已生成         != waveform 已通过播放/质量验收
```

模型、协议、执行器、声码器和 verifier 都要单独记录。

## 91.3 AuT：用音频时间轴进入语言模型

AuT 是 Qwen3-Omni 的音频编码器。报告公开的主线包括：约 2,000 万小时监督音频、8 倍 Conv2D 下采样、12.5 Hz 音频 token rate、动态约 1-8 秒 attention window，以及约 0.6B 参数规模。数据构成还区分中英文伪标注 ASR、其他语言 ASR 和音频理解数据。

12.5 Hz 可以换算为：

```text
每个音频时间步 ~= 1 / 12.5 s = 80 ms
```

这不是端到端首包延迟。真实首包路径至少包括音频读取/解码、AuT 编码、Thinker chunked prefill、工具或安全门禁、Talker 首码本、Code2Wav、网络和播放器缓冲。任何“80 ms 所以 80 ms 出声”的回答都把 token 粒度误当成系统延迟。

## 91.4 TM-RoPE：统一时间和空间坐标

Qwen3-Omni 使用 temporal/height/width 三个位置维度的 TM-RoPE。报告给出的公开细节包括：rotary angle 分配 `24/20/20`，音频 temporal ID 以 80 ms 为粒度，视频按真实 timestamp 对齐到 80 ms。

它解决的是多媒体位置表示问题：

```text
audio timestamp  ----\
video timestamp  ----- > temporal axis (80 ms grid)
image height/width ---/       + spatial axes
```

和固定 2 秒切块相比，真实 timestamp 让模型可以表达任意时长流式输入中的跨 chunk 时间关系。但 serving 仍需保存媒体时间戳、采样/丢帧策略、chunk 边界、token offset 和 cache handle。TM-RoPE 不能自动解决长视频检索、跨 chunk 状态恢复或网络乱序。

## 91.5 Talker：首码本、MTP 和 Code2Wav

语音 codec 使用多码本 residual vector quantization。Talker 的生成路径可以抽象为：

1. 自回归路径预测第一个 codebook，承担主要序列决策。
2. MTP 预测剩余 residual codebooks，减少完全串行生成的深度。
3. 以 12.5 Hz 的音频 code rate 继续流式输出。
4. 轻量 causal ConvNet 的 Code2Wav 将 code 转成 waveform。

因此 Talker 的优化目标至少有三层：首码本的语义/韵律质量、残差码本的重建质量，以及 Code2Wav 的实时率和跨 packet 连续性。MTP 只说明码本预测的并行化方向，不能直接推出所有 GPU、batch、音频长度上的 acceptance rate 或实时率。

## 91.6 多阶段预训练与后训练

报告描述的预训练可以简化为：

```text
S1: freeze LLM, train encoder/adapter
  -> S2: ~2T mixed text/audio/image/video/video-audio tokens
  -> S3: 8K -> 32K context, more long audio/video
```

Thinker 后训练包含 lightweight SFT、strong-to-weak distillation、off-policy/on-policy KL、GSPO 和多模态 rule-based/model-based reward。Talker 后训练包含多模态语音 continual pretraining、long-context、多语言 DPO 和 speaker fine-tuning。

这里的关键不是背流程名，而是知道每步修复什么问题：S1 先对齐 encoder/adapter，S2 建立跨媒体理解，S3 让上下文分布覆盖长媒体；Thinker 后训练主要塑造指令/推理/工具行为，Talker 后训练还要处理声音质量、多语言和说话人控制。论文没有公开完整 reward 权重、teacher 数量或所有训练 kernel，因此不能把流程名扩写成完整 recipe。

## 91.7 异步 chunked prefill 与高并发

Thinker 和 Talker 的生成节奏不同：Thinker 需要处理多模态理解、推理与可能的工具回合，Talker 需要连续发出音频 code 和 waveform packet。官方资料描述异步 chunked prefill，目的在于把长输入分块并减少高并发下的整体阻塞。

一个服务调度器至少要区分：

| 状态 | 需要记录的字段 |
|---|---|
| 多模态输入 | media revision、timestamp、采样策略、视觉布局 |
| Thinker | chunk offset、attention mask、工具/策略事件、文本输出 |
| Talker | codebook index、residual state、packet sequence、取消点 |
| Code2Wav | chunk 边界、音频格式、重叠/拼接状态 |
| 评测 | TTFT、首音频包、首视频包、TPOT、p95、失败恢复 |

报告给出的理论 first audio packet `234 ms`、first video packet `547 ms` 是论文/发布方数字，不是本机实测。一次严谨的 benchmark 还要固定媒体、并发、GPU、后端、网络、工具、缓存、取消和播放端缓冲。

## 91.8 开源运行入口和部署边界

官方 README 当前给出 `transformers>=5.2.0`、`qwen-omni-utils` 和 FlashAttention 2 的依赖；`disable_talker()` 可节省约 10GB 显存。README 还说明 vLLM 主要支持 Thinker，Instruct 的音频输出仍处于实现推进阶段。

因此部署状态应分层：

```text
repository loads
  -> Thinker local inference
  -> audio encoder/vision preprocessing
  -> Talker code generation
  -> Code2Wav waveform
  -> backend streaming
  -> target hardware + tool/verifier acceptance
```

“仓库能运行”不等于“完整多模态流式输出能运行”；“Thinker 可以服务”也不等于“vLLM 已经支持 Instruct 音频输出”。面试时要主动说出未验证层，而不是把安装命令当成生产证明。

## 91.9 代码练习：用时间戳合并跨媒体事件

下面的零依赖示例只演示 TM-RoPE 相关的时间栅格化和事件排序。它不是 Qwen3-Omni 的 encoder、RoPE kernel 或生产 packet scheduler。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    kind: str
    timestamp_ms: int
    value: str


def temporal_id(timestamp_ms: int, step_ms: int = 80) -> int:
    if timestamp_ms < 0:
        raise ValueError("timestamp must be non-negative")
    return timestamp_ms // step_ms


def merge_events(events: list[Event]) -> list[tuple[int, Event]]:
    return sorted(
        ((temporal_id(event.timestamp_ms), event) for event in events),
        key=lambda item: (item[0], item[1].timestamp_ms, item[1].kind),
    )


stream = [
    Event("video", 161, "frame-b"),
    Event("audio", 80, "speech-a"),
    Event("video", 79, "frame-a"),
]

for token_id, event in merge_events(stream):
    print(token_id, event.kind, event.timestamp_ms, event.value)
```

输出会把 79 ms 和 80 ms 映射到相邻的时间 bin。生产实现还必须处理 timestamp jitter、视频帧重复/丢失、音频重采样、chunk overlap、跨 batch offset 和真正的 rotary embedding；这个例子只帮助面试时解释“真实时间对齐”和“端到端流式调度”不是同一个问题。

## 91.10 面试追问

**问：为什么 Talker 不只读取 Thinker 的文本？**

答：报告描述 Talker 使用音频/视觉多模态特征。这样可以保留更直接的跨模态语音生成，同时允许 Thinker 链路接入 RAG、function calling 和安全过滤器。它增加了状态同步和调度复杂度，但避免把所有信息先压成文本再生成声音。

**问：MTP 为什么能降低语音生成延迟？**

答：第一个 codebook 仍按自回归路径生成，剩余 residual codebooks 由 MTP 预测，因此码本维度上的串行深度下降。最终延迟还取决于 code rate、接受/校验策略、Code2Wav、GPU kernel 和 packet 拼接，不能只看 MTP 名称。

**问：TM-RoPE 和普通三维位置编码有什么区别？**

答：它把 temporal、height、width 分开，并把音频和视频时间对齐到约 80 ms 的真实时间尺度；重点是跨媒体时间语义，而不是简单给 token 加一个序号。仍需保存 timestamp 和 chunk 状态，位置编码不能代替流式状态管理。

**问：234 ms 的首音频包是否说明线上服务 p99 小于 234 ms？**

答：不是。那是报告口径的理论/实验数字。线上还要加入输入解码、排队、chunked prefill、工具/安全门禁、声码器、网络和播放器缓冲，并按目标硬件和并发测 p50/p95/p99。

**问：Qwen3-Omni 当前能否直接用 vLLM 输出语音？**

答：官方 README 当前说明 vLLM 主要支持 Thinker，Instruct 音频输出仍在实现推进中。应把模型代码、音频预处理、Talker/Code2Wav、后端 streaming 和目标硬件 acceptance 分开验收。

## 91.11 小练习

1. 画出 Thinker、RAG/function calling/safety、Talker、Code2Wav 和播放器之间的状态流，标出哪些边界由模型负责、哪些由宿主负责。
2. 用合成 audio/video timestamp 实现 80 ms 时间栅格，分别报告跨 chunk 丢帧、重复帧和 jitter 对事件顺序的影响。
3. 为多码本语音生成设计 latency manifest，分别记录首码本、residual codebooks、Code2Wav、网络 packet 和播放缓冲耗时。
4. 设计一个四级 serving gate：Thinker inference、Talker code、waveform、目标硬件/工具/verifier acceptance；每级写出失败时是否可重试。
5. 比较 `disable_talker()` 前后的显存、首 token、文本质量和语音能力，说明“节省显存”不能直接等价为“端到端更快”。

## 91.12 来源

- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Qwen3-Omni Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [Qwen3-Omni Thinking 配置](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking/raw/main/config.json)
- [Qwen 官方博客](https://qwen.ai/blog?id=65f766fc2dcba7905c1cb69cc4cab90e94126bf4&from=research.latest-advancements-list)
- [阿里云 Qwen-Omni API 文档](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni)
- 完整证据与待核验项：[`qwen3-omni-source-notes.md`](../../research/model-update-2026-09/qwen3-omni-source-notes.md)
