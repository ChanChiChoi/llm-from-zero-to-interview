# 语音识别、音频生成与实时对话来源

检索与阅读日期：2026-10-29；论文版本页面复核：2026-10-30；Emformer 论文补充核阅：2026-11-05。

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

### Emformer

- 来源：[Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition](https://arxiv.org/abs/2010.10759v4)，arXiv:2010.10759v4，2020-12-30。
- 阅读范围：摘要、§2.2（记忆库与左上下文 Key/Value 缓存）、§3.2 的算法延迟及 hybrid/transducer 结果。
- 来源支持：模型把长程历史压缩到增强 memory bank，缓存左上下文注意力的 Key/Value，并用并行分块训练支持低延迟流式模型。作者在 LibriSpeech 报告平均算法延迟约 960 ms 时，test-clean/test-other WER 分别为 2.50%/5.62%；80 ms 设置分别为 3.01%/7.09%。
- 证据等级：source-backed（论文架构、LibriSpeech 数据、作者报告的算法延迟和 WER）。
- 边界：延迟与 WER 对应论文的模型、解码器、数据切分和测量口径；不证明普通话电话、口音/噪声切片、专名或订单字段准确率，也不等于客户端端到端首声延迟。

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

## 2026-11-02 流式语音结构与服务测量

### Qwen3.5-Omni 的 ARIA

- 来源：[Qwen3.5-Omni Technical Report](https://arxiv.org/html/2604.15804)，arXiv:2604.15804v2（2026-04-21）。访问日期 2026-11-02；阅读摘要与 §2.5 “Designs for Streaming and Concurrency” 中的 “Streaming Generation with ARIA”。
- 来源支持：作者将流式语音合成的不稳定/不自然与文本和语音 tokenizer 的编码效率差异联系起来，并提出 ARIA 动态对齐文本/语音单元；§2.5 说明 ARIA 将 Qwen3-Omni 的双通道生成改写为受单调交织约束的单一 text/speech stream，目标是降低两条生成轨道的同步开销并适配增量解码。报告将其 TTFT（Thinker 首文本 token）、TTFC（Talker 首音频 chunk）与 Overall Latency（首个可播放包关键路径）分别定义；理论延迟数据使用作者内部 vLLM、`torch.compile` 和 MTP/codec decoder CUDA Graph 设置。
- 证据等级：source-backed（作者报告的结构说明与测量定义）。
- 边界：tokenizer 编码效率差异是作者对该类流式问题的解释和模型动机，不是对所有 TTS 的因果证明；性能口径受 Qwen3.5-Omni、内部 serving、并发和加速配置限制，不能据理论首包推断客户端网络 TTFA。

### vLLM-Omni 多阶段语音 serving

- 来源：[vLLM-Omni Technical Report](https://arxiv.org/html/2610.09307)，arXiv:2610.09307v1（2026-10-07）。访问日期 2026-11-02；阅读 §2 的 orchestration/stage/connector 职责、§5.1 测量口径、Figure 9/Table 6 与 TTS Table 8。
- 来源支持：报告把 Thinker → Talker → Code2Wav 组织为多阶段服务，异步 chunk 路径允许下游读取部分 codec 输出；作者区分 Thinker 首文本 token TTFT 与听者首音频包 TTFP。在 Qwen3-Omni Fixed-Len-2500/900、两张 H100、并发 32 的 Table 6 中，async-chunk/no-async-chunk 的均值分别为 TTFT 507.8/3420.1 ms、E2EL 72.63/136.62 s、audio RTF 0.630/0.927。TTS Table 8 在 NVIDIA H200、Seed-TTS English texts 条件下报告 Qwen3-TTS CustomVoice 并发 8 吞吐 81.1/111.6 audio-s/s；默认双 GPU 与融合单 GPU profile 的并发 1 中位 TTFP 为 21.5/15.3 ms。
- 证据等级：source-backed（预印本架构、指标定义和指定 runtime recipe 的实验）。
- 边界：报告的 H100/H200 硬件、nightly freeze、固定文本/输出长度、并发、profile 和版本提交均影响结果；TTFT 不等于 TTFP，论文吞吐也不能外推到其他 TTS、网络或质量门槛。数值未本地复现。

### NVIDIA Riva 流式 chunk/padding 示例

- 来源：[Building and Deploying ASR Pipelines](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/public/asr/asr-pipeline-configuration.html)，NVIDIA Riva 动态官方文档。访问日期 2026-11-02；阅读 Conformer `StreamingRecognize` 低延迟和高吞吐配置段。
- 来源支持：当前页面为低延迟流式配置列出 `chunk_size=0.16`、`padding_size=1.92`；高吞吐流式配置列出 `chunk_size=0.8`、`padding_size=1.6`。低延迟示例还包含相应 tokenizer、verbalizer、decoder 与 endpointing 参数，故该组合不是独立模型常数。
- 证据等级：source-backed（访问日 NVIDIA 官方配置文档）。
- 边界：配置绑定 Riva 的 Conformer 流式 API 与文档所述 pipeline；把 `chunk+2×padding` 当作输入窗口只是在左右 padding 对称时的估算，不能直接推出 GPU 工作量、并发或目标识别质量。动态页面在未来部署前应按目标 Riva 版本复查。

## 2026 年长文、场景语音与音频注入研究

检索与阅读日期：2026-11-12。

### SwanBench-Speech

- 来源：Pan et al., [Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](https://arxiv.org/abs/2605.28618v1)，arXiv:2605.28618v1，2026-05-27。
- 阅读范围：摘要、§2 长文评估基准表、§3.1 基准范围、§3.4 指标、§3.5 人类感知对齐摘要、§4 评估设置、§5 讨论及附录 E 实验环境/语音条件说明。
- 来源支持：SwanBench-Speech 覆盖长篇 TTS 和对话生成，共 1,101 个样本、17 种场景；指标分为声学、语义和表现力三组，共七项，覆盖音色/混响一致性、声音保真、内容准确、韵律连贯、表现丰富度和表现层次。作者对部分自动指标进行听者研究，并报告长篇、高表现力条件仍有持续性差距。
- 证据等级：source-backed（预印本基准构造、指标定义和作者评测）。
- 边界：结论绑定所选场景、数据、参考音频、推理配置和自动评估器；自动表现力打分采用 Gemini 3 Pro 等特定 judge，不能直接外推为其他语种/产品表现或普遍模型排名。该来源适合说明评测单位和指标维度扩展，不证明七项分数可以组成一个通用总分。

### DRAMA-Eval

- 来源：Hu et al., [DRAMA-Eval: A Benchmark for Context-Aware, Multi-Turn, Multi-Speaker Speech Generation](https://arxiv.org/abs/2610.31796v1)，arXiv:2610.31796v1，2026-10-22。
- 阅读范围：摘要、§3.1–3.3 任务和数据、§4.1–4.5 推理协议/客观与自动评估/人类一致性/全局上下文消融、§6 limitations。
- 来源支持：数据包含 1,038 个中英文案例（中文 537、英文 501），源音频 12.50 小时、标注语音 10.65 小时和 7,530 个轮次；每案例平均 43.36 秒、7.25 轮，涉及 1–4 位说话人。协议区分 WER/CER 与跨轮说话人一致性等客观指标、逐轮指令/韵律/表现力，以及场景交互/整体流畅/沉浸度。后六项使用 Gemini 3.8 Flash 与真实音频做 A/B 比较，报告 tie-adjusted win rate，并提供人工一致性分析。
- 证据等级：source-backed（预印本数据说明、评分协议、人工校验和作者实验）。
- 边界：只含中文与英语，自动 judge 是专有音频语言模型；不同模型类别通过各自支持的不同输入条件评测；平均场景约 43 秒，不等同长篇叙事；不评估背景音乐、环境音和音效。模型排名须绑定类别与接口，不能把场景评分直接解释成通用语音质量。

### AudioHijack 与第三方音频注入

- 来源：Chen et al., [Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection](https://arxiv.org/abs/2604.14604v1)，arXiv:2604.14604v1，2026-04-16。
- 阅读范围：摘要、§III 威胁模型、§V 评估与隐蔽性、§VI 商业代理实验、§VII 对策、§VIII 限制及伦理说明。
- 来源支持：论文把 threat model 定义为第三方只能提供/影响音频数据、不能控制用户其它上下文，并提出 AudioHijack 攻击；作者报告在其 13 个测试 LALM、六类行为和未见上下文实验下的攻击成功，并在若干商业语音代理官方 API 上演示工具行为操纵。论文同时显示单靠 prompt 级告知或 response self-reflection 在其设置下并不足以可靠检测。
- 证据等级：source-backed（预印本攻击定义、受控模型评估和商业 API 研究实验）。
- 边界：预印本不是任意录音、模型、声学环境或生产集成系统的攻击率；作者指出其测试平台、模型版本、攻击类别和代理集成覆盖有限。该结果支持把音频理解与用户授权、工具权限和副作用确认分开，不支持把报告中的攻击率外推成目标产品风险概率。

## 音频质量评测：MUSHRA 与 FAD

检索与阅读日期：2026-11-12。

### MUSHRA

- 来源：[ITU-R BS.1534-3](https://www.itu.int/rec/r-rec-bs.1534/en)，《Method for the subjective assessment of intermediate quality level of audio systems》，2015-10，访问时状态为 In force (Main)。
- 阅读范围：ITU 推荐页面与 BS.1534-3 官方 PDF 的范围、实验设计、听者筛选、刺激呈现、评分、统计分析和报告章节。
- 来源支持：MUSHRA 是针对中等音质水平音频系统的多刺激主观评测方法，要求设计隐藏参考与锚点，并规定听者、测试素材、聆听条件、统计分析和结果报告。
- 证据等级：verified（现行 ITU-R 推荐文本）。
- 边界：它是受控主观评测协议，不会自动回答语音内容是否正确、说话人是否获授权、实时交互是否可靠；用于 TTS 或其他任务时仍须定义属性、样本和比较条件。

### Fréchet Audio Distance

- 来源：Kilgour et al., [Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms](https://arxiv.org/pdf/1812.08466v4)，arXiv:1812.08466v4，2019-01-17。
- 阅读范围：摘要、引言、§3.1–3.2 的定义与 VGGish 嵌入、§4 的实验设置/数据、§5 的失真和人工比较结果。
- 来源支持：原始 FAD 面向音乐增强，不要求逐样本配对参考；VGGish 将 1 秒音频窗转换为 128 维嵌入，作者以 0.5 秒步长取重叠窗口，分别估计待评输出集合和干净背景音乐集合的多元高斯统计，再计算 Fréchet 距离。论文在 MagnaTagATune 数据（600 小时；540 小时背景集、60 小时评估集）上构造人工失真，并在 300 段 5 秒素材的听测子集上比较指标；作者报告该实验中 FAD 与人类偏好的相关系数为 0.52，高于所比较的 SDR、余弦距离和幅度 L2。
- 证据等级：source-backed（原论文 v4 的定义、数据和指定人工听测）。
- 边界：FAD 是评估集合相对背景分布的统计量，受嵌入模型、参考背景集合、样本量和预处理影响；论文的音乐增强结果不能外推为通用 TTS 自然度、内容准确率、授权或说话人安全指标。后续不同 embedding、sample-bias 修正或任务定义属于不同评估配置，应单独说明。
