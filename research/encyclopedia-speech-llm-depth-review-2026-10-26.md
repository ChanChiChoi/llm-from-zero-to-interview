# 第四册第十八章《语音大模型》本轮深度复核（2026-10-26）

## 复核范围

- 对象：`book-04-llm-encyclopedia/chapters/18-语音大模型.md`，依据当前 `WRITING_SPEC.md` 对 18.0—18.23 复读并增补。
- 旧稿保留：波形、Mel、encoder、ASR/WER/CER、关键字段、Whisper、speech token、neural codec、TTS、韵律、声纹/克隆、speech-to-speech、流式 ASR、VAD、endpointing、turn-taking/barge-in、端到端模型、语音 Agent、安全、端到端评估、成本账本代码和原六项练习均保留。
- 本轮策略：没有拆文件；把研究补充集中在最常被独立追问的监督/对齐/生成/实时交互路线，原有工程和安全主线留在本章。
- 背景：2026-10-23 有旧轮次的格式、数值和账本代码检查记录。本次不将旧记录算作本轮覆盖；新研究记录和 `coverage_map` 在此文件。

## 核心问题图谱与代表路线

| 问题 | 核心方案 | 实际解决的问题 | 主要代价/失败边界 | 本章去向 |
| --- | --- | --- | --- | --- |
| 转写标注不足，encoder 怎么获得声学表示？ | wav2vec 2.0、HuBERT、Whisper | 分别利用 latent 对比学习、聚类伪标签 masked prediction、互联网弱转写多任务学习减少对高质量逐句标注语料的依赖 | 前两者下游通常仍需标注/迁移；弱监督标签会错配、偏向高资源语言；实验数据集/预训练规模不可横比 | 18.3 新增监督目标对照；18.5 深化 Whisper 数据、task token 和边界 |
| 音频帧比文本长，怎样做无逐帧标注的流式识别？ | CTC、RNN-T、attention seq2seq | CTC 对所有可折叠路径求和；RNN-T 在时间与文本前缀二维格点上学习转导；attention seq2seq 自回归利用上下文 | CTC 帧级条件独立假设简单但依赖 decoder/LM 补语言上下文；RNN-T 搜索和服务更复杂；seq2seq 常需要 chunk/单调注意力才能流式 | 18.13 新增机制、公式和受控实验边界 |
| 离散语音码要保留什么？ | semantic token、acoustic codec token、语义蒸馏 codec | 语义单元偏内容鲁棒；codec 码偏声音可重建；Mimi 把 SSL 高层语义蒸馏入 RVQ 第一层码，尝试避免两者简单拼接 | 降低声学细节会损伤韵律/身份；保留细节会带来 token rate、噪声和隐私问题 | 18.7 增补选择标准；18.8 说明 token 总量和串行步数不是一回事 |
| TTS 如何处理同一文本的多种合法读法？ | Tacotron 2、FastSpeech 2、VITS、VALL-E/VALL-E 2 | 分别代表 Mel+vocoder、自回归并行化和显式方差条件、VAE/flow/adversarial 联合生成、codec LM 的 zero-shot 说话人条件 | AR 对齐/延迟；并行模型依赖 aligner 与 variance predictors；联合端到端损失较难诊断；codec LM 受码率/数据覆盖影响且有重复环路和身份冒用风险 | 18.9 新增路线比较；18.10 的韵律控制与 one-to-many 问题在该表衔接 |
| 实时对话系统如何缩短反应并允许打断/重叠？ | cascade、Moshi、多阶段 Thinker-Talker、ARIA | cascade 保留阶段审计；Moshi 并行建模双方音流；Thinker-Talker 分离多模态理解和音频生成；ARIA 处理 text/codec token rate mismatch | 端到端架构减少显式模块边界但提高归因/安全难度；理论首包、服务器时间、实际播放首音和不同并发口径不可横比 | 18.16 增加代表工作比较；18.18 区分 TTFT/TTFC/首包/首音；详细 ARIA 见第二十一册第 93 章 |
| 系统是否满足真实任务？ | WER/CER + critical-field accuracy + MOS/preference + TTFA/RTF + task success | 将内容准确、可懂度、自然度、实时性、动作安全和成功成本分别测量 | 指标依赖分母、评测器/听者、端点和输入切片；单一 MOS/WER/首包不能代表上线质量 | 18.18 增加指标定义和自动评估风险；18.20 保留成本账本 |

## 旧知识点覆盖图

| 原章节 | 本轮处理 | 解释层级/位置 |
| --- | --- | --- |
| 18.0—18.2：连续波形、采样与 log-Mel | 保留并深化输入信息和定义域；本轮研究确认无需改写原推导 | 核心基础，连续机制/参数边界/数值例子保留 |
| 18.3：CNN/RNN/Transformer/Conformer 与 encoder 表示 | 保留网络结构表；新增 wav2vec 2.0/HuBERT/Whisper 预训练目标对照 | 核心：分清 encoder 架构与预训练监督目标，记录不同数据依赖 |
| 18.4：ASR、WER/CER、critical fields | 完整保留 | 核心评估口径，既有分母、空集、规范化与字段例子保留 |
| 18.5—18.6：Whisper 与转录后处理 | 扩展 Whisper 数据量、任务 token、30 秒窗口和论文报告的 long-form 失败 | 核心方案；本轮无证据支持的目标业务性能不作外推 |
| 18.7—18.8：speech token、codec、rate | 保留四种旧表示并补 semantic/acoustic/mixed 的目标冲突；解释 codec ID 数与 AR 解码步不等价 | 核心生成/序列成本；Moshi/Qwen 只作分层预测例子，不重复其整章架构 |
| 18.9—18.10：TTS 与韵律 | 保留原 pipeline 和维度，新增 Tacotron 2、FastSpeech 2、VITS、VALL-E/2 机制/收益/限制对照 | 核心方法谱系；各论文 MOS 绑定自己数据集和听评协议 |
| 18.11：speaker embedding / voice cloning | 保留授权、身份和隐私论述 | 核心安全与适用边界 |
| 18.12：cascade 与 speech-to-speech | 保留原对照，新增代表性架构证据和适用边界 | 核心架构比较，强调信息瓶颈、latency 与可追溯性 |
| 18.13：streaming ASR | 增加 CTC 路径求和公式、RNN-T 二维转导和 attention seq2seq 缓冲边界 | 核心机制；受控研究发现仅用于 7 语言/指定同 encoder 配方 |
| 18.14—18.15：VAD、endpoint、turn-taking、barge-in | 保留状态机、延迟账本、取消旧请求与工具动作的关系 | 核心系统状态，不以本轮论文路线覆盖 |
| 18.16：端到端语音模型 | 保留原四类结构，将代表方案扩为 cascade/Moshi/Qwen3-Omni/Qwen3.5-Omni+ARIA | 核心研究路线；区分 full-duplex、streaming output 与模块化双工 |
| 18.17：Multimodal Voice Agent | 保留工具权限、trace、失败链和确认边界 | 核心工程/安全，链接至本章其他章节不迁移 |
| 18.18：端到端评估 | 保留链式事件分解，补 MOS/ASR proxy/TTFT/TTFC/首音/RTF 的测量边界 | 核心评测；避免跨论文分数和系统组件时间直接相加 |
| 18.19—18.20：隐私、安全、单位成功成本、toy ledger | 完整保留；代码不改 | 核心治理/工程；教学账本仍不冒充生产成本或模型 benchmark |
| 18.21—18.23：延伸阅读、六项练习、小结 | 保留原入口和六项练习，新增 ACL 综述、离散语音 token 综述、CTC/RNN-T 实验与路线选型练习 | 支持原主线迁移到方案选择，不删除旧内容 |

## 来源与证据

本轮以 2025 ACL SpeechLM 综述建立组件地图，以原始论文和技术报告核机制/数据，再用已有专门 Qwen3.5-Omni 章节避免重复铺开。检索日期：2026-10-26。

| 主题 | 来源与本章使用的事实 | 证据限制 |
| --- | --- | --- |
| 综述地图 | [Recent Advances in Speech Language Models（ACL 2025）](https://aclanthology.org/2025.acl-long.682/) 归纳 tokenizer、LM、vocoder、训练阶段、交互范式和评测；[Recent Advances in Discrete Speech Tokens](https://arxiv.org/abs/2502.06490) 区分语义/声学 token、语义蒸馏、长度压缩与流式挑战 | 只作为分类/问题入口，不据综述替代论文机制或效果证据 |
| 自监督/弱监督预训练 | [wav2vec 2.0](https://arxiv.org/abs/2006.11477)：latent masking + quantized contrastive targets；53K h unlabeled + 10min labeled 的 LibriSpeech 结果；[HuBERT](https://arxiv.org/abs/2106.07447)：cluster pseudo-label masked prediction、迭代聚类；[Whisper](https://arxiv.org/abs/2212.04356)：680K h 弱监督，117K h 覆盖其他 96 语种，125K h 翻译样本，统一多任务 decoder | 数字不横向比较；Whisper 作者指出弱标签错配、低资源语种不均、long-form hallucination/repetition/漏首尾；这些是论文所测任务与时期的证据 |
| 流式识别对齐目标 | [Benchmarking LF-MMI, CTC and RNN-T](https://arxiv.org/abs/2011.04785)：同一 streamable encoder、7 个语言/3K—14K h 数据，报告 RNN-T 精度优势与 CTC 推理效率优势；论文为不同目标分别调 units/stride，未含 attention seq2seq | 不推广成各类 ASR 任务的普遍排名；机制解释并不表示目标硬件 latency |
| TTS 路线 | [Tacotron 2](https://arxiv.org/abs/1712.05884)：attention character→Mel + WaveNet vocoder；[FastSpeech 2](https://arxiv.org/abs/2006.04558)：ground-truth Mel、duration/pitch/energy、LJSpeech 约 24 h 单说话人、20 native English listeners、MOS 3.83±0.08；[VITS](https://proceedings.mlr.press/v139/kim21f.html)：conditional VAE/flows/adversarial/stochastic duration；[VALL-E](https://arxiv.org/abs/2301.02111)：60K h、3 s enrollment、codec LM、论文自述口音/朗读风格覆盖限制；[VALL-E 2](https://arxiv.org/abs/2406.05370)：repetition-aware sampling 与 grouped code | MOS 均由各论文自己的数据/听者/系统条件得出，不横排；VALL-E 2 “human parity”严格限定其 LibriSpeech/VCTK 实验，不作为普遍人类质量判断 |
| 全双工与流式架构 | [Moshi](https://arxiv.org/abs/2410.00037)：parallel user/assistant streams、Mimi semantic-acoustic RVQ、Inner Monologue、论文的 160ms theoretical/200ms practical；[Qwen3-Omni](https://arxiv.org/abs/2509.17765)：Thinker-Talker、MTP residual codes、causal ConvNet Code2Wav、报告 234ms theoretical cold-start first-packet；[Qwen3.5-Omni v2](https://arxiv.org/abs/2604.15804v2)：ARIA 自适应 text/speech rate constraint、MTP/streaming codec，内部 vLLM 报告 Flash audio 235ms / Plus 435ms | 厂商/作者报告，硬件、并发、chunk/packet/端点定义不同；Qwen3.5 报告明确 Flash/Plus 资源配置不同，不作严格横向比较；更多 ARIA 机制和 training/eval gaps 见本地第 93 章与 source note |

## 配套同步

- `PAPERS.md`：加入 ACL 2025 SpeechLM 综述、离散语音 token 综述、CTC/RNN-T 对照、VALL-E 2 和 Moshi；Qwen3-Omni/Qwen3.5-Omni 的深资料已在该文件有专题条目。
- `INTERVIEW_BANK.md`：扩写 ASR/TTS/Speech-to-Speech、Whisper、codec LM 的题目；加入 CTC/RNN-T/attention seq2seq 的机制对比题，并将后续同区题号顺延。
- `GLOSSARY_EN_ZH.md`：补 CTC、RNN-T、attention seq2seq、semantic/acoustic speech token、TTFA、TTFC、RTF、ARIA 的术语定位。
- 第四册目录无需改；本章没有拆分。Qwen3.5-Omni 已链接到第二十一册第 93 章。

## 静态复核与未完成边界

- `formula_status`：静态检查 CTC path-sum 中 blank/重复标签折叠含义、codec `DqK` 数量与自回归 schedule 的区别、现有 WER/CER/关键字段/RTF/延迟/任务成功率定义；修正 CTC 的连续重复字示例为 `b, o, blank, o, k → book`。尚未做外部公式渲染。
- `code_status`：本轮没有改写原有账本代码，未运行代码示例或测试。旧版 2026-10-23 研究记录中的代码执行状态不作为本轮运行证据。
- `link_status`：新增 arXiv、ACL Anthology、PMLR 原始论文链接；已有内部第 93 章相对链接应通过相对路径检查。
- `validation`：须做 Markdown/围栏/章节标题与 `git diff --check` 静态检查；未执行 ASR/TTS 模型、端点、PDF 或真实语音 demo。
- `open_risks`：未在目标语言/噪声/设备下复现论文；未部署全双工语音系统或测真实用户可听首音；未复测声纹误识、克隆授权、水印鲁棒性、ASR proxy 偏差或高影响工具确认；不同论文延迟/MOS/WER不能横向排名。Qwen3.5-Omni 的详细机制仍以发布方报告和第 93 章为准，未经本机权重或 API probe。
- `status`：第四册第十八章完成本轮来源驱动的深度复核与章节级静态验收；全书 goal 继续进行，下一份为第四册第十九章。
