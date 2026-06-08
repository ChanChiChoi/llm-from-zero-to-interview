# 项目路线

## 当前状态

24 本主书正文第一版已完成，第三册已完成项目作品集和多模态实战。本文件用于把可写进简历的项目统一整理成路线、产出物和面试表达。第二轮精修时，本文件需要继续补齐 Agent Harness、工具协议生态、AI Infra 和推理框架方向的项目任务。

## 项目 1：miniGPT from scratch

目标：从零实现 decoder-only Transformer，并完成小语料自回归语言建模实验。

简历表达：基于 PyTorch 实现 miniGPT，包括 tokenizer、tensor shape / dtype / device / stride 审计、autograd / backward / zero_grad / detach / no_grad 审计、nn.Module / Parameter / buffer / state_dict / train-eval / save-load 审计、Dataset / DataLoader / collate_fn / sampler / padding waste 审计、训练循环 / 梯度累积 / 梯度裁剪 / scheduler / eval / checkpoint resume 审计、mixed precision / autocast / GradScaler / activation checkpointing / OOM audit 审计、DDP rank / world size / all-reduce / DistributedSampler / global batch / no_sync / rank 0 checkpoint 审计、debug/profiling metadata / finite / hook / grad norm / timing / profiler 审计、causal attention、Transformer block、验证 loss、perplexity、序列 log probability 和采样生成，并分析重复生成、过拟合、概率高不等于事实正确、上下文长度限制、分布式训练同步风险和性能瓶颈归因。

## 项目 2：手写 LLM 核心组件

目标：实现 tokenizer、PyTorch tensor shape / dtype / device / broadcasting / stride 审计、PyTorch autograd / backward / grad accumulation / detach / no_grad 审计、PyTorch nn.Module / Parameter / buffer / state_dict / train-eval / save-load 审计、PyTorch Dataset / DataLoader / collate_fn / sampler / worker 审计、PyTorch training loop / optimizer step / scheduler step / eval loop / checkpoint resume 审计、PyTorch mixed precision / AMP / memory / OOM 审计、PyTorch distributed training / DDP / all-reduce / global batch / no_sync 审计、PyTorch debug / profiling / finite / hook / timing / profiler 审计、PyTorch Transformer component / RMSNorm / MHA / RoPE / KV Cache 审计、线性代数 shape 审计、概率/NLL/PPL 审计、信息论 loss / KL / MI 审计、优化器与学习率调度审计、SVD / PCA / LoRA 低秩审计、泛化与过拟合审计、贝叶斯与校准审计、RLHF / PPO / DPO 数学审计、评估统计与显著性审计、数学面试复盘审计、attention、causal mask、multi-head attention、RoPE、top-k、top-p、KV Cache 和 DPO/CLIP loss。

简历表达：手写并验证大模型关键训练与推理组件，能解释 PyTorch tensor metadata、device mismatch、broadcasting、view / reshape / contiguous、padding / causal mask、LM loss flatten、autograd dynamic graph、leaf / non-leaf tensor、vector-Jacobian product、grad accumulation、zero_grad、detach、no_grad、in-place 风险、missing grad 排查、nn.Module 注册树、Parameter 与普通 tensor、ModuleList / ModuleDict、buffer 与 state_dict、train/eval、strict loading、冻结参数与 optimizer 参数组、保存加载一致性、Dataset / DataLoader 职责边界、map-style / iterable-style dataset、collate_fn 动态 padding、labels ignore index、length bucket、DistributedSampler、worker seed、pin memory、padding waste 和 data pipeline audit、training loop step 顺序、raw / scaled loss、gradient accumulation、gradient clipping、optimizer step、scheduler step、validation token-weighted loss、checkpoint resume、non-finite loss 和 training loop audit、FP32 / FP16 / BF16 显存估算、AMP autocast、GradScaler、loss scaling / unscale、activation checkpointing 重算、CUDA memory stats、OOM audit、DDP rank / local rank / world size、process group、all-reduce 平均梯度、global batch、DistributedSampler 切分、no_sync 通信优化、collective deadlock audit、rank 0 checkpoint、debug tensor metadata、finite check、forward hook、gradient debug、CUDA synchronize timing、torch.profiler、DataLoader bottleneck 和 debug/profiling audit、Transformer component audit、RMSNorm、scaled dot-product attention、multi-head self-attention、SwiGLU MLP、Pre-Norm decoder block、RoPE、LM head weight tying、KV Cache decode mask、dot product、cosine similarity、矩阵乘法 shape、attention score、softmax 概率分布、entropy、cross entropy、KL、mutual information、NLL/PPL、SGD / Momentum / AdamW、warmup + cosine、gradient clipping、global batch、SVD rank-r 误差、PCA explained variance、LoRA 参数量、train / validation / test 泛化 gap、bias-variance、early stopping、distribution shift、污染 overlap、Bayes update、MAP、Beta-Bernoulli 可靠率、ECE、Brier score、answer / verify / abstain 路由、MDP、discounted return、value、advantage、PPO ratio、clipped surrogate、RLHF KL penalty、reward model pairwise loss、DPO loss、mean / variance / standard error、confidence interval、paired bootstrap、McNemar test、sample size、Bonferroni / Benjamini-Hochberg 多重比较校正、实验上线 guardrail、数学面试 topic coverage / formula accuracy / demo coverage / weak questions / revision plan、top-k/top-p 采样、每个模块的输入输出 shape、复杂度和工程坑。

扩展要求：补充 PyTorch engineering interview readiness 模块，用 0 依赖脚本审计回答是否覆盖 tensor、autograd、Module、DataLoader、training loop、AMP / memory、DDP、debug / profiling 和 Transformer components，并输出 topic coverage、formula checks、debug coverage、red flags、revision plan 和 gate pass，作为项目面试前的自检报告。

## 项目 3：LoRA/SFT 中文助手

目标：使用开源小模型完成中文 instruction tuning。

简历表达：构建中文 SFT 数据集，正确处理 chat template 与 assistant-only loss mask，使用 PEFT/LoRA 完成监督微调，并评估 base vs SFT 的指令遵循、格式合法率、人工偏好和 bad case。

## 项目 4：DPO 偏好优化实验

目标：构造 chosen/rejected preference pair，使用 TRL/LoRA 完成 DPO 偏好优化。

简历表达：基于中文 SFT 助手构造偏好数据，覆盖事实性、完整性、格式遵循、安全拒答和拒答边界，比较 SFT 与 DPO 的 pairwise 胜率、输出长度、安全拒答和通用能力回归。

## 项目 5：高性能推理服务

目标：部署开源 LLM，支持 Chat API、SSE 流式输出、限流、监控和压测。

简历表达：构建大模型推理服务，测量 TTFT、P95/P99、tokens/s、QPS、显存和错误率，对比 Transformers generate、vLLM、量化和不同并发下的性能差异。

## 项目 6：企业知识库 RAG 系统

目标：完成文档解析、清洗、chunk、embedding、混合召回、rerank、引用生成、权限过滤和评估。

简历表达：构建可评估的企业知识库问答系统，使用 Recall@K、MRR、groundedness、citation accuracy 和人工评测分析检索与生成质量，并分类复盘解析、切分、召回、重排、生成和权限类 bad cases。

## 项目 7：数学推理增强实验

目标：构造数学推理评测集，比较 zero-shot、few-shot、CoT、self-consistency 和工具校验。

简历表达：实现答案抽取与标准化、self-consistency 投票、Python 表达式校验和错误类型分析，按题型统计准确率、成本和 bad cases。

## 项目 8：CLIP 图文检索

目标：使用 CLIP 实现 text-to-image 和 image-to-text 检索，并手写 mini CLIP loss。

简历表达：使用 image/text encoder 生成归一化 embedding，通过 cosine similarity 和 FAISS 实现图文检索，手写 `N x N` 相似度矩阵与双向 CLIP loss，使用 Recall@K、MRR、Top-1 Accuracy 评估，并分析中文 prompt、temperature / logit scale、细粒度识别和 OCR 场景局限。

扩展要求：补一个 0 依赖 toy demo，验证 L2 normalize、CLIP 双向 loss、text-to-image / image-to-text rank、zero-shot prompt ensemble 和 bad case 分类表。

## 项目 9：开源 VLM 图片问答

目标：使用 LLaVA、Qwen-VL、InternVL 或 MiniCPM-V 跑通图片输入、文本问题和多模态回答流程。

简历表达：构造包含描述、OCR、图表、截图、计数和空间关系的小型 VQA 评测集，分析 vision encoder 分辨率、patch size、CLS / patch tokens、视觉幻觉、OCR 错误、视觉 token 预算、上下文占用和 prompt 对结果的影响。

扩展要求：实现一个 0 依赖多模态成本审计脚本，输入图片分辨率、patch size、视频帧数、音频时长、文本 token 和上下文上限，输出 image / video / audio token 估算、over budget 样本和上线门禁。

扩展要求：实现一个 vision encoder shape audit，检查 `[B,C,H,W] -> [B,N,d_v] -> [B,N,d_l]` 的 patch grid、position embedding 长度、projector shape、attention cell 数和 OCR / 图表高分辨率取舍。

扩展要求：实现一个 VLM connector / prompt shape audit，检查 image placeholder 数量是否等于图片数量、projector 输出是否匹配 LLM hidden size、直接拼接多图是否超过上下文上限、Q-Former / Perceiver Resampler 压缩后是否回到预算内，以及 assistant-only label mask 是否只覆盖回答 token。

## 项目 10：多模态 Instruction Tuning 数据集

目标：构造 image-text conversation 数据，理解 image token、chat template 和 label mask。

简历表达：设计多模态 SFT 样本格式，覆盖单图问答、多轮对话、OCR、图表理解和拒答边界，并说明 assistant-only loss mask、图片占位符一致性、视觉 token 数和截断策略。

扩展要求：实现一个多模态 SFT data audit，输入 toy image-text conversation 样本，输出 placeholder mismatch、context over budget、label mask error、unsupported answer、missing refusal、task coverage、evidence support rate 和 rejected sample 列表。

## 项目 10A：Diffusion 基础审计

目标：实现一个教学版 diffusion 加噪 / 去噪审计脚本，理解 DDPM 的 forward diffusion、noise prediction、scheduler、CFG 和 latent diffusion 成本直觉。

简历表达：手写 DDPM toy demo，计算 `beta_t`、`alpha_t`、`alpha_bar_t`、闭式加噪 `x_t`、噪声预测 MSE、`x0_hat`、DDPM reverse mean、classifier-free guidance 和 latent / pixel cost ratio，并解释采样步数、scheduler、guidance scale 和 latent compression 对质量、速度和稳定性的影响。

扩展要求：输出一张 diffusion audit report，包含 alpha schedule 是否单调、`x_t` shape、noise MSE、reconstruction MAE、CFG guided prediction、latent ratio、sampling calls 和 gate pass。

## 项目 10B：Stable Diffusion / DALL-E pipeline 审计

目标：实现一个教学版文生图 pipeline 审计脚本，理解 Stable Diffusion 的 VAE、text encoder、denoiser、scheduler、CFG、ControlNet 和 DALL-E 自回归图像 token 路线的成本差异。

简历表达：手写 0 依赖 text-to-image pipeline audit，输入 prompt、negative prompt、分辨率、VAE scale、采样步数、CFG scale、ControlNet 条件和 DALL-E image token 网格，输出 latent shape、VAE 压缩比、U-Net 调用次数、cross-attention cells、image-to-image 起始步、ControlNet condition shape、自回归图像 token 数和上线 gate，并解释 latent diffusion、negative prompt、ControlNet 和自回归图像 token 的 trade-off。

扩展要求：输出一张文生图审计报告，包含 prompt token 截断风险、CFG 额外 denoiser 调用、scheduler / seed 可复现性、ControlNet shape 对齐、inpainting mask 对齐、安全过滤和版权风险检查项。

## 项目 10C：视频生成与 World Model 审计

目标：实现一个教学版视频 token / 时序一致性审计脚本，理解视频生成的时空 token 成本、video diffusion 推理成本、temporal consistency、world model rollout 和安全门禁。

简历表达：手写 0 依赖 video generation audit，输入帧数、分辨率、spatial patch、temporal patch、latent 压缩倍数、采样步数、CFG scale 和 toy object trace，输出 framewise tokens、spatiotemporal tokens、latent video shape、latent ratio、denoiser calls、cross-attention cells、flicker、identity stability、motion smoothness、object permanence、world rollout MAE 和 gate pass，并解释视频生成相对图像生成在时序一致性、物理合理性、长程依赖和安全风险上的 trade-off。

扩展要求：输出一张视频生成审计报告，包含 token budget、latent shape、采样成本、FVD / 人评指标、identity drift、flicker、物理失败、deepfake / 名人肖像 / 虚假新闻 / provenance 风险和上线门禁。

## 项目 11：图像 + 文本 RAG

目标：把 OCR、图片描述、文本检索、图像检索和 LLM 生成结合起来。

简历表达：构建多模态 RAG demo，将图片 OCR、caption、metadata 与文本知识库统一索引，支持图文混合检索和基于证据的答案生成。

## 项目 12：语音转文本再问答 Demo

目标：串联 Whisper ASR、LLM 问答和文本/TTS 输出。

简历表达：实现语音输入到文本转写、问题理解、知识库问答和结果输出的端到端 demo，并分析 ASR 错误如何影响下游问答。

## 项目 12A：音频 token / codec / WER 审计

目标：实现一个教学版音频 token 审计脚本，理解 waveform、log-mel spectrogram、ASR WER / CER、audio codec、codec language model、TTS token 预算和 cascade latency。

简历表达：手写 0 依赖 audio generation audit，输入采样率、音频时长、window / hop、mel bins、reference / hypothesis、codec 帧率、codebooks、TTS 时长、级联延迟和说话人授权状态，输出采样点数、log-mel shape、WER、CER、codec token 数、压缩比、TTS token 数、codec LM context、ASR / LLM / TTS 级联延迟和 gate pass，并解释语音系统在准确率、自然度、实时性、codec token 成本和 voice cloning 安全之间的 trade-off。

扩展要求：输出一张语音上线审计报告，包含 ASR 分场景 WER / CER、噪声鲁棒性、VAD 切片、流式首包延迟、TTS MOS / 说话人相似度、codec token budget、voice cloning 授权、声音水印、深伪检测和隐私合规检查项。

## 项目 12B：统一多模态助手 token / routing 审计

目标：实现一个教学版统一多模态样本审计脚本，理解统一多模态系统的 token budget、routing、loss mixture、Any-to-Any 输出预算和安全门禁。

简历表达：手写 0 依赖 unified multimodal audit，输入文本 token、图片分辨率、音频时长、视频帧数、输出文本 / 音频 / 图像 token 预算、上下文上限、loss weights、任务需求和安全标记，输出 text / image / audio / video / special token、输入 / 输出总 token、attention cells、decoder cells、路由模块、loss mixture、risk flags 和 gate pass，并解释接口统一、表示统一、early-fusion token 统一、中心 LLM 系统和 Any-to-Any 模型在成本、能力、可控性和安全上的 trade-off。

扩展要求：输出一张统一多模态上线审计报告，包含 OCR / ASR / video sampler / retriever / image generator / TTS / tool executor 路由正确性、上下文预算、输出预算、跨模态冲突、不可信媒体指令隔离、voice cloning 授权、生成输出审核、工具权限和多任务 loss mixture 覆盖情况。

## 项目 12C：多模态评估与安全审计

目标：实现一个教学版多模态评估与安全审计脚本，理解 VQA、OCR / ASR、图表、grounding、faithfulness、prompt injection、隐私、身份安全和上线门禁。

简历表达：手写 0 依赖 multimodal evaluation and safety audit，输入 toy VQA 人工答案、OCR / ASR reference 和 hypothesis、图表预测值、grounding bbox、claim support 标注和安全策略样本，输出 VQA accuracy、OCR WER / CER、ASR WER、chart relaxed accuracy、mean IoU、precision@0.5、supported claim rate、hallucination rate、policy accuracy、risk counts、P95 latency gate 和 `gate_pass`，并解释公开 benchmark、分布指标、人评、证据支持和安全门禁之间的 trade-off。

扩展要求：输出一张多模态上线评估报告，包含 clean task success、图文 / 音文 / 视频证据支持、OCR / ASR 分场景错误、图表数值容忍、grounding IoU、图像 / 视频生成 FID / FVD 参考、deepfake / voice cloning / 生物身份风险、不可信媒体指令、隐私脱敏、内容 provenance、over-refusal、latency 和事故回溯字段。

## 项目 13：大模型评估平台 Mini Version

目标：管理 prompt set、模型版本、自动评估、人工评估、LLM-as-Judge 和 bad case 回归。

简历表达：构建可复现的大模型评估和回归测试工具，支持版本对比、pairwise 评测、硬门禁、评测报告和 bad case 管理。

## 项目 14：Coding Agent Harness Mini Version

目标：实现一个最小 coding agent runtime，支持任务输入、文件读写、命令执行、权限检查、trace 记录和 replay。

简历表达：设计并实现 coding agent harness，包含 planner、tool executor、workspace patch、shell sandbox、permission gate、trace/replay 和失败复盘机制，能够分析 Claude Code、OpenCode、Codex 类系统的工程取舍。

## 项目 15：工具协议与 MCP Server Demo

目标：实现一个最小工具协议 demo，覆盖 tool schema、参数校验、权限边界、MCP server 接入和工具调用评估。

简历表达：构建企业工具调用 demo，支持结构化 function calling、工具注册、参数 schema 校验、权限控制、调用 trace 和错误处理，并设计工具选择、参数正确率、执行成功率和安全性的评估表。

## 项目 16：AI Infra 可观测性与成本 Dashboard

目标：模拟训练和推理平台的核心指标，构建可观测性、容量规划和成本分析 dashboard。

简历表达：设计 AI Infra dashboard，覆盖 GPU 利用率、MFU/HFU、队列等待、训练故障、TTFT、TPOT、tokens/s、KV Cache、P95/P99、错误率和单位 token 成本，并能用指标定位训练和推理瓶颈。

## 项目 17：Mini LLM Serving Engine

目标：从 naive generate loop 升级到支持 continuous batching、KV Cache 管理、prefix cache、流式输出和简单调度的教学版 serving engine。

简历表达：实现 mini LLM serving engine，包含 request queue、prefill/decode 分离、token budget 调度、KV block 管理、流式返回和压测脚本，并对比 naive generate、vLLM/SGLang 思路下的延迟、吞吐和显存 trade-off。

## 项目 18：Web-Scale 数据采集审计 Mini Version

目标：构建一个不联网的教学版数据采集审计工具，模拟 source registry、HTML 解析、质量评分、规则过滤、exact / near dedup、MinHash / SimHash 候选召回、code / math / domain 专项数据审计、synthetic / distillation data 生成审计、preference / safety data 审计、multimodal data 审计、data attribution / valuation、dataset versioning / governance、PII / 密钥扫描、安全隔离、评估污染检测、canary 命中、data mixture 采样权重、effective epoch、配比统计、数据版本报告和数据工程面试自评。

简历表达：实现 web-scale 数据采集审计 mini pipeline，能解释 license / ToS / robots / provenance、teacher authorization、prompt registry、synthetic ratio、preference rubric、safety taxonomy、image-text alignment、OCR layout、ASR / WER、video temporal alignment、grounded multimodal QA、source-level ablation、gradient similarity、Data Shapley 近似、negative value data、manifest、lineage、deletion request、datasheet 和 model card 的工程边界，输出质量分、exact duplicate group、near duplicate cluster、code license / secret / test 审计、math answer / verifier 审计、domain authority / PII 审计、synthetic / distillation 授权与验证审计、chosen / rejected margin、标注一致性、长度偏置、误拒 / 漏拒修复覆盖、图文 / 音文 / 视频对齐审计、视觉 / 音频隐私审计、数据源价值排名、attribution proxy、token budget 下的数据选择、dataset manifest、shard checksum、lineage coverage、访问阻断、删除请求命中、train-eval overlap、canary 命中、sampling weight、effective epoch、保留率、语言和领域配比、风险命中率、拒绝原因、误删 / 漏检审计口径、版本审计报告和 mock interview readiness report，并说明如何从 toy pipeline 扩展到真实训练数据治理和数据工程面试表达。

## 项目 19：Safety Alignment Eval、Goal Mismatch Audit、Reward Hacking Audit、Scalable Oversight 与 Guardrail Demo

目标：构建一个不联网的教学版安全评估、目标错配审计、reward hacking 审计、可扩展监督、jailbreak / prompt injection 防护审计、red teaming / dangerous capability 审计、mechanistic interpretability 审计、steering / representation engineering 审计和护栏 demo，模拟 risk taxonomy、policy expected action、proxy objective、outer mismatch、behavior mismatch、proxy follow、Goodhart gap、reward-human gap、proxy mismatch、reward hacking rate、length bias、high-reward-low-quality ratio、goal misgeneralization、human direct coverage、AI feedback accuracy、verifier coverage、process step accuracy、evidence support、high-risk audit coverage、instruction hierarchy、untrusted content boundary、prompt injection success、indirect injection success、data leakage、unauthorized tool call、attack task success、clean task success、red team taxonomy coverage、P0/P1 unresolved、capability elicitation gain、dangerous capability uplift、autonomy score、red team regression pass rate、activation patch recovery、ablation effect、SAE reconstruction fidelity、feature purity、interpretability gate、steering vector、projection shift、target uplift、safe gain、side-effect drop、over-refusal delta、steering gate、unsafe compliance、refusal accuracy、over-refusal、attack success、safe completion quality、severity-weighted risk、tool permission gate、audit log、prompt injection gate、red team gate、reward hacking gate、oversight gate、alignment gate、release gate 和 safety release gate。

扩展要求：补充 model editing / unlearning 审计模块，模拟 toy edit cases、paraphrase checks、locality checks、retain drop、forget set、retain set、改写 / 多轮泄露、membership inference 风险、editing gate 和 unlearning gate，并在项目报告中解释“目标样本改对”“输出抑制”和“近似遗忘证据”之间的区别。

扩展要求：补充 privacy / watermarking 审计模块，模拟 toy PII / secret / memorization / RAG / logging / membership / watermark cases，输出 PII recall、secret recall、memorization rate、output leak rate、RAG unauthorized retrieval、raw log rate、membership advantage、watermark z-score、generated recall、false positive rate、robust recall、privacy gate 和 watermark gate。

扩展要求：补充 governance / model card 审计模块，模拟 toy model card sections、system card sections、policy categories、eval slices、risk issues、approval votes 和 release checks，输出 model card completion、system card completion、policy coverage、eval coverage、severity-weighted unresolved risk、high-risk mitigation、approval coverage、governance gate 和 release ready。

扩展要求：补充 safety interview retrospective 模块，模拟 toy answer records、risk set、metric set、gate set 和 trade-off set，输出 answer coverage、risk coverage、metric coverage、gate coverage、trade-off coverage、unsafe detail count、readiness gates 和 revision plan。

简历表达：实现安全对齐评估 mini pipeline，能把 helpful、honest、harmless、alignment problem、reward hacking、scalable oversight、jailbreak、prompt injection、red teaming、dangerous capability、mechanistic interpretability、steering、model editing、unlearning、privacy、watermarking、governance 和 safety interview readiness 的冲突转成可量化指标和上线门禁，覆盖真实目标、代理目标、模型行为、部署行为、reward model / LLM judge / benchmark proxy、RAG 证据核查、代码 / 数学 verifier、Agent trace、高风险请求、边界允许请求、隐私、直接 / 间接提示注入、工具结果注入、工具调用、capability elicitation、危险能力 baseline uplift、red team regression、activation patching、SAE feature 审计、steering 强度扫描、model editing 局部性检查、unlearning forget / retain set、PII / secret / memorization / RAG / log 审计、水印检测、model card / system card 完整度、安全面试回答覆盖度和 over-refusal 场景；输出风险分类分布、outer mismatch、behavior mismatch、proxy follow、Goodhart gap、reward-human gap、reward hacking rate、高 reward 低质量样本、goal misgeneralization、AI feedback 校准、verifier 覆盖、过程监督准确率、证据支持率、高风险人审覆盖、instruction hierarchy violation、prompt injection success、data leakage、unauthorized tool call、attack task success、clean task success、boundary coverage、P0/P1 unresolved、dangerous capability uplift、autonomy score、regression pass rate、patch recovery、ablation effect、reconstruction fidelity、feature purity、steering vector、target uplift、safe gain、side-effect drop、over-refusal delta、edit success、generalization、locality、retain drop、forget leak、robust leak、membership risk drop、PII recall、secret recall、memorization rate、output leak rate、watermark z-score、generated recall、false positive rate、policy coverage、eval coverage、severity-weighted unresolved risk、approval coverage、answer coverage、gate coverage、trade-off coverage、漏拒 / 误拒、对抗成功、工具越权、安全替代质量、严重度加权风险、失败切片和修复建议，并说明模型本体对齐、policy layer、工具权限、不可信内容边界、红队回归、mechanistic interpretability、representation engineering、model editing、unlearning、privacy gate、watermark gate、governance gate、prompt injection gate、red team gate、steering gate、editing gate、unlearning gate、reward hacking gate、scalable oversight、release gate、监控、事故响应和面试复盘如何组成安全闭环。
