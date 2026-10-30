# 语音识别、音频生成与实时对话来源

检索与阅读日期：2026-10-29；论文版本页面复核：2026-10-30。

本记录只存放联网核验得到的技术事实及其证据范围，不记录章节审阅、修改决定或任务进度。

## 语音识别与对齐

### Whisper

- 来源：[Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/html/2212.04356)，arXiv:2212.04356v1，2022-12-06。
- 阅读范围：§2.1–2.3（数据处理、模型与多任务格式）、§3.8（长音频转写）、§4.5（长音频解码策略）及附录训练数据统计。
- 来源支持：模型输入采样为 16 kHz，使用 80 通道 log-Mel 表示，论文报告 25 ms 窗口和 10 ms 帧移；训练语料约 680,000 小时，作者另报告 96 种其他语言约 117,000 小时和约 125,000 小时跨语言到英语翻译数据。长音频以 30 秒窗口处理并使用时间戳滑窗及解码策略。
- 边界：这些数字属于论文的训练数据分类，不能在未证明互斥时相加重算总量。模型评测结论受论文 checkpoint、数据集、任务 token、解码与文本规范化约定限制；不证明任意语言、噪声、实时产品或目标业务上的准确率。

### CTC

- 来源：[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)，ICML 2006 原论文。
- 阅读范围：方法与训练目标部分、TIMIT 实验及网络结构说明。
- 来源支持：CTC 通过 blank 与重复标签折叠定义输入帧路径到输出标签序列的映射，并对映射到同一目标序列的路径概率求和，训练时不要求逐帧标签对齐。原论文实验使用双向 LSTM。
- 边界：CTC 是对齐/训练目标，不能单独推出编码器因果性或流式服务行为。双向 LSTM 实验说明使用 CTC 不自动代表在线识别；低延迟还取决于 encoder 上下文、分块、lookahead、解码和 endpoint 策略。

### RNN-T

- 来源：[Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/html/1211.3711)，arXiv:1211.3711v1，2012-11-14。
- 阅读范围：摘要、§2.1–2.6（prediction/transcription network、输出分布、forward-backward、训练与测试）。
- 来源支持：方法引入 null/blank 输出；含 blank 的对齐路径在移除 blank 后映射到输出序列，并在这些可能路径上求和。模型由转录网络与预测网络构成，可联合依赖输入表示和输出前缀；作者实验是 TIMIT 音素识别。
- 边界：论文提出一般序列转导目标，不构成现代 ASR 运行时延迟或跨模型准确率排名证据。是否能低延迟输出，还要看输入网络是否可增量计算、可见未来上下文、解码与 endpoint 设计。

## 音频 codec 与生成

### SoundStream

- 来源：[SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/html/2107.03312)，arXiv:2107.03312v1，2021-07-07。
- 阅读范围：摘要、§III 模型、§IV 评测设置、§V 结果与 §VI 结论。
- 来源支持：作者提出卷积 encoder/decoder 与 residual vector quantizer 的端到端 codec；论文报告通过 quantizer 层的 structured dropout 支持 3–18 kbps 可变码率，并在作者测试设置中实现可流式的低延迟推理。
- 边界：码率、质量和实时性是论文配置与实验平台下的报告，不是任意 codec、设备或语音生成模型的保证。

### EnCodec

- 来源：[High Fidelity Neural Audio Compression](https://arxiv.org/html/2210.13438)，arXiv:2210.13438v1，2022-10-24。
- 阅读范围：摘要、§3 模型/量化、§4.3–4.6 评测与结果、§4.6 延迟和计算时间。
- 来源支持：论文介绍流式 encoder-decoder 和量化 latent 表示，并报告对 codec 输出进行 Transformer entropy modeling 的压缩实验；作者使用 MUSHRA 主观测试评估特定带宽和音频域。
- 边界：MUSHRA 结果属于论文所用听者、刺激、带宽和音频域；不能把 codec 音质分数当成 TTS 自然度或说话人相似度，也不能跨不一致协议直接比较。

### Moshi

- 来源：[Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037)，arXiv:2410.00037v1，2024。
- 阅读范围：§3.4 多流音频建模与推理，§5.7 streaming ASR/TTS，以及摘要的实时交互说明。
- 来源支持：论文描述用户与助手并行音频流的全双工 speech-text 系统，并报告该模型及其推理设置下的流式指标。
- 边界：论文中的理论或实测延迟只支持其模型、音频 tokenization、chunk 和硬件/推理设置；不能外推成一般产品延迟。并行流是系统接口例子，不等于所有语音模型都能处理重叠发言。

### VALL-E 2

- 来源：[VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers](https://arxiv.org/html/2406.05370)，arXiv:2406.05370v2，2024-06-17。
- 阅读范围：§3 分组 codec 语言建模与推理，§4.1 评测设置、§4.2 LibriSpeech 和 §4.3 VCTK 主客观评测。
- 来源支持：作者提出 repetition-aware sampling 和 grouped codec modeling，并报告在 LibriSpeech、VCTK 的指定鲁棒性、自然度和说话人相似度实验中达到其所述 human-parity 结果。
- 边界：human parity 结论严格限定于论文 benchmark、数据、主观评价与比较协议；不能解释为所有语言、提示音频、句子类型或部署场景都与人类语音等价。

## 测量边界

- WER/CER 的教学公式需和文本规范化、分词/字符单位、参考分母和聚合方式一起解释。逐句宏平均与先汇总错误数再除以参考总量采用不同权重；空参考分母应单列。此处属于指标定义和数学口径，不代表某个 benchmark 的统一 harness 实现。
- 实时语音的端到端延迟须固定事件起点与终点。模型产出首个音频块、客户端播放首音、首个稳定转写和用户打断后完全停止旧音频是不同观测点；流式阶段重叠时，完整阶段时长之和只是串行近似，不能替代真实关键路径测量。

## 2026 年 ASR 架构与业务评估

### Qwen3-ASR

- 来源：[Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v2)，arXiv:2601.21337v2，2026-01-30。
- 阅读范围：摘要、§1、§2.1—2.3、表 1。
- 来源支持：报告描述 Qwen3-ASR-0.6B/1.7B 与 Qwen3-ForcedAligner-0.6B。ASR 架构以 AuT attention-based encoder-decoder 为声学前端，对 128 维 Fbank 做 8 倍下采样，使用动态 1—8 秒 attention window；作者称这套实现支持 streaming 与 offline 两种推理。ForcedAligner 另提供 word/sentence/paragraph 等可变粒度时间戳预测。报告还把流式能力描述为具体 encoder window、chunk 与推理系统的组合。
- 证据等级：source-backed（作者技术报告的模型架构和公开评测描述）。
- 边界：52 种语言/方言、首 token latency、吞吐和 benchmark 结论绑定对应 checkpoint、版本、数据、硬件、窗口和评测协议；不能据模型属于 AED 就推断所有 AED 都可流式，也不能据论文 latency 直接承诺客户端 TTFA。

### mu-bench：对业务语音输入评价语义错误

- 来源：[mu-bench: A Multilingual Utterance Transcription Benchmark](https://arxiv.org/html/2609.32082v1)，arXiv:2609.32082v1，2026-09-25。
- 阅读范围：摘要、§2.1—2.2、§3.1—3.3、§4.1—4.4、§5.1—5.6、§6 和伦理说明。
- 来源支持：数据包含 250 通电话中的 4,270 条来电 utterance，涵盖 en-US、es-MX、tr-TR、vi-VN、zh-CN，主要测试 AI 银行业务流程里姓名、联系方式、日期、金额和确认码等字段输入。作者提出 reference-guided language-aware normalization 和 Utterance Error Rate（UER）LLM judge。在其 1,847 条人工评分样本、指定 judge/normalizer 设置下，UER 与人工“重大错误/非重大错误”的 Cohen’s kappa 为 0.779；normalized exact-match WER 为 0.533。作者也报告归一化会把 263 条人工标为重大错误中的 12 条改写成参考文本，说明规范器可能掩盖真实错误。
- 证据等级：source-backed（作者预印本、数据集设计与人工评审协议）。
- 边界：数据限于五种 locale、电话采样的虚构银行对话；UER 依赖 judge、rubric、normalizer 和人工标注。它补充业务语义切片，不替代 WER/CER，也不构成跨 provider 或通用 ASR 排名。

## 2026 年 TTS 与韵律评估

### Qwen3-TTS

- 来源：[Qwen3-TTS Technical Report](https://arxiv.org/html/2601.15621v1)，arXiv:2601.15621v1，2026-01-22。
- 阅读范围：摘要、§1、§2.1—2.2、§3.1。
- 来源支持：报告描述 25 Hz 单码本路线，配合分块 Flow Matching DiT 与 BigVGAN 解码；另有 12.5 Hz 多码本 semantic/acoustic 表示、双轨文本/语音 token 架构和 MTP，配合因果 Code2Wav 解码。作者报告的 97 ms first-packet latency 属于其 Qwen3-TTS-0.6B 版本和报告推理设置。
- 证据等级：source-backed（作者技术报告中的架构、tokenizer 和作者测量）。
- 边界：模型家族名称、码率/质量、公开模型能力和首包延迟受具体 checkpoint、chunk、decoder、硬件和测量端点限制；不能直接解释为端到端客户端 TTFA 或所有 TTS 的排序。

### Long-Prompt Zero-Shot TTS 与 LP-TTS-Bench

- 来源：[Long-Prompt Zero-Shot TTS: Benchmarking Prosody Reproduction Beyond Voice Cloning](https://arxiv.org/html/2610.19774v1)，arXiv:2610.19774v1，2026-10-14。
- 阅读范围：摘要、§2.1—2.2、§3.1、§4.1—4.4、§5 和伦理说明。
- 来源支持：论文把长语音 prompt 下的 speaker-specific prosody reproduction 作为独立任务，发布双语 37-speaker LP-TTS-Bench（17 位中文说话人、20 位英语说话人；63.95 小时 prompt，2,728 条 held-out test utterance）。协议除 intelligibility 和 timbre similarity 外，还评估 timing、prominence、intonation 与 prosody 表示。作者的 prompt-duration 消融发现多数韵律指标整体在约 10 分钟附近较好，之后收益趋于饱和且部分指标变化并非单调。
- 证据等级：source-backed（2026 年 arXiv v1 预印本及其公开数据/评分方案说明）。
- 边界：结果绑定论文中的 Qwen3TTS-based LPTTS、双语 37 人基准和特定自动/主观指标；预印本将代码、音频与 benchmark 公开时间写为录用后。它支持把 timbre 和 prosody 分开评估，不证明所有 zero-shot TTS 都受益于长 prompt。

## 2026-10-30 Transformers/Whisper 接口复核

- 来源：[Transformers v5.17.0 Whisper 文档](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/whisper)、[v5.17.0 pipeline 文档](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/pipelines)、[v5.17.0 ASR pipeline 源码](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/pipelines/automatic_speech_recognition.py)、[当前 ASR pipeline 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py)、[OpenAI Whisper v20250625 音频实现](https://github.com/openai/whisper/blob/v20250625/whisper/audio.py)、[长音频转写实现](https://github.com/openai/whisper/blob/v20250625/whisper/transcribe.py)、[Whisper 发布记录](https://github.com/openai/whisper/releases)、[openai/whisper-small 的 revision 973afd2](https://huggingface.co/openai/whisper-small/commit/973afd2)。
- 阅读范围：Transformers Whisper 文档“Longform transcription”与 pipeline 输出说明；v5.17.0 和当前 ASR pipeline 的 `chunk_length_s` 警告、预处理和 Whisper 时间戳后处理；Whisper 固定发布版本中的 30 秒窗口常量、长音频 seek/时间戳循环；模型仓库提交记录。
- 来源支持：Transformers 文档要求长于 30 秒的音频不截断，将完整 log-Mel 特征交给 Whisper 自定义 `generate()` 并开启 `return_timestamps=True`；长音频由基于时间戳的顺序解码及压缩率、平均 log probability、温度回退等启发式处理。v5.17.0 与当前 pipeline 源码都把对 seq2seq 模型使用通用 `chunk_length_s` 标为实验性；针对 Whisper，源码建议用模型自己的 `generate()` 长音频路径。pipeline 时间戳返回形状为 `text` 与可选 `chunks`，每项含 `text` 和 `timestamp`；Whisper 词级时间戳由 cross-attention 权重结合动态时间规整（DTW）近似得到，不是独立强制对齐器的精确词边界。`openai-whisper` 的 20250625 是核对日可见的官方发布版本；模型仓库 revision `973afd2` 于 2024-02-29 提交 tokenizer merge 更新。
- 证据等级：source-backed（版本化官方文档、官方仓库固定 release 与模型仓库提交记录；current pipeline `main` 源码按访问日读取）。
- 边界：v5.17.0 文档和 tagged pipeline 源码用于固定示例 API；`main` pipeline 源码是动态页面，未来行为应以目标安装版本复查。Whisper revision 只固定模型仓库文件，不能替代 Transformers、PyTorch、音频解码器或硬件版本锁定。未据此推断准确率、时间戳精度或实时性能。
