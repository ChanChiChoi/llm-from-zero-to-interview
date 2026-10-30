# 多模态系统路线与音视频评测：来源证据

检索日期：2026-10-29。本记录只保存联网核验的新知识点、来源范围和证据边界；书籍审阅、纳入与修改决定记录在 `progress/`。

## 视觉编码器与语言模型的连接路线

- Liu et al., “Visual Instruction Tuning,” arXiv:2304.08485, [论文](https://arxiv.org/abs/2304.08485)。实际阅读摘要和方法/训练部分，重点核对 §3 的 projection matrix 与 §4 的两阶段训练。
  - 来源支持：LLaVA 将视觉编码器输出用可训练矩阵投影到语言模型 embedding 维度，形成视觉 token；第一阶段冻结视觉编码器和语言模型、训练投影矩阵；第二阶段继续冻结视觉编码器并联合微调投影矩阵与语言模型。论文 §3 描述用 GPT-4 生成视觉指令数据；§4 给出 CC3M 特征对齐和多模态指令微调阶段。
  - 证据等级：source-backed（原始论文的方法定义与该模型实验）。
  - 边界：这是 LLaVA 论文中的具体设计；不能据此断言所有 VLM 都使用单层线性 projector，或其接口会让所有视觉特征都被正确使用。GPT-4 合成数据的质量、覆盖和偏差不由生成流程本身保证。

- Li et al., “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models,” NeurIPS 2023 / arXiv:2301.12597 v3, [论文](https://arxiv.org/abs/2301.12597)。实际阅读摘要、引言及 §3.1–3.3。
  - 来源支持：BLIP-2 以可训练的 Querying Transformer（Q-Former）桥接冻结图像编码器和冻结语言模型；论文分两阶段训练视觉—语言表示与视觉条件生成。生成阶段用全连接层把 query 表示投影到 LLM embedding 维度并以前缀 soft visual prompt 接入。论文实验使用 32 个 query，并举出冻结 ViT-L/14 特征 `257×1024` 与 query 输出 `32×768` 的对照。
  - 证据等级：source-backed（原始论文的方法定义和指定实验）。
  - 边界：257→32 表示序列位置减少 225 个、约压缩 8.03 倍（相对减少约 87.55%），仅是论文给定表示的长度计算；不是端到端 FLOPs、延迟或总成本比例。Q-Former 仍需读取视觉特征，可能存在细节信息瓶颈。query 数、编码器和实验收益属于论文配置，不是所有 query-based 连接器的默认行为或比较保证。

- Alayrac et al., “Flamingo: a Visual Language Model for Few-Shot Learning,” arXiv:2204.14198, [论文](https://arxiv.org/abs/2204.14198)。实际阅读摘要、§3.1.1–3.1.2。
  - 来源支持：Flamingo 用 Perceiver Resampler 将变长视觉特征映射为固定数量的视觉 token，并在冻结语言模型层之间插入新的 gated cross-attention blocks 来读取视觉特征。
  - 证据等级：source-backed（原始论文的架构方法说明）。
  - 边界：这是一条特定的图文/视频 few-shot 架构路线；固定 token 数和跨注意力位置不能外推到其他模型。

## 公开模型与产品的证据边界

- [Qwen3.5-397B-A17B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)，访问日期 2026-10-29。实际阅读 `Qwen3.5 Highlights` 中 Unified Vision-Language Foundation 段。
  - 来源支持：发布方将该模型描述为 unified vision-language foundation，并称其采用 multimodal tokens 的 early-fusion training。
  - 证据等级：source-backed（发布方的可变模型卡/产品资料）。
  - 边界：模型卡是可变页面和发布方自述，不公开完整 tokenization、全部模块连接、训练配方或独立复现实验；不能据此推断其每个请求的内部路径，也不能把“early fusion”泛化为所有原生多模态模型的统一结构。

- OpenAI, [“Hello GPT-4o”](https://openai.com/index/hello-gpt-4o/) 与 [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)，2024-05-13。实际阅读发布说明和 system card 的摘要/模型说明。
  - 来源支持：发布材料称 GPT-4o 可接受文本、音频、图像、视频的组合输入并产生文本、音频、图像的组合输出；system card 将其描述为 autoregressive omni model。
  - 证据等级：source-backed（供应商发布说明和 system card）。
  - 边界：这类材料支持特定模型的公开能力声明，不公开全部 tokenizer、routing、训练数据、模态间中间接口和线上服务实现；发布演示也不等于独立性能测量。

## Transformers 的图像问答接口

- 来源：[Transformers 图像到文本任务指南](https://huggingface.co/docs/transformers/tasks/image_text_to_text)、[多模态 chat template 文档](https://huggingface.co/docs/transformers/chat_templating_multimodal)、[Qwen2.5-VL 模型文档](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) 与 [Qwen2.5-VL-3B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)，访问日期 2026-10-30。
- 阅读位置：多模态消息内容格式、`AutoProcessor.apply_chat_template()` 的 `tokenize`/`return_dict`/`return_tensors` 参数、生成输入字段；Qwen2.5-VL 的模型类和 processor 示例。
- 来源支持的事实：当前 Qwen2.5-VL 文档使用 `Qwen2_5_VLForConditionalGeneration` 和 `AutoProcessor`；`apply_chat_template()` 可一并准备图文消息并生成供模型调用的张量输入。通用任务页及模型卡还展示自动模型类写法，具体类名和字段随 Transformers / checkpoint 接口演进。
- 证据等级与边界：source-backed（访问时的官方库文档和发布方模型卡）。动态文档不保证特定依赖组合已联调；文档代码不证明本地运行、显存需求、质量或延迟。需要复现时应固定模型与 processor revision、Transformers/PyTorch/Accelerate 版本和图像内容。

## 音视频评测候选

- Yang et al., “OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning,” arXiv:2610.12458, [论文](https://arxiv.org/abs/2610.12458)，2026-10-08 提交。实际阅读 arXiv 摘要和版本信息；未通读全文。
  - 来源支持：摘要报告 786 段密集标注视频，并把输出拆成实体指代、视觉镜头和音频事件等可核验单元，用于诊断时间定位、身份漂移、跨模态错配和幻觉描述。
  - 证据等级：preprint / source-backed（预印本摘要描述的 benchmark 设计和数据集规模）。
  - 边界：摘要不足以核对全部标注协议、筛选条件、judge 一致性和实测效应；规模与错误类型绑定该数据集和评测设计，不能代表一般视频理解或生产多模态系统。

## 视觉问答中的对象幻觉、图像上下文诊断与注入风险

- Li et al., “Evaluating Object Hallucination in Large Vision-Language Models,” arXiv:2305.10355, [论文](https://arxiv.org/abs/2305.10355)，访问日期 2026-10-29。实际阅读摘要、§3.1 与 POPE 方法部分。
  - 来源支持：POPE 将对象是否存在转成是/否二分类轮询问题，并通过不同负样本采样方式构造探针；作者指出既有对象幻觉评估可能受指令和生成风格影响。
  - 证据等级：source-backed（原始论文提出的协议和论文实验）。
  - 边界：它聚焦对象存在判断和特定样本/轮询设置，不覆盖所有视觉幻觉、区域 grounding 或任意生产问答；其模型排序不应脱离论文数据与解析设置外推。

- Guan et al., “HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models,” CVPR 2024 / arXiv:2310.14566, [论文](https://arxiv.org/abs/2310.14566)，访问日期 2026-10-29。实际阅读摘要、§3 的问题分类与编辑图像/无图对照示例。
  - 来源支持：论文以 346 张图像和 1,129 个问题组织图像上下文推理评估，区分视觉依赖与视觉补充类问题，并用相关对照分析语言幻觉、视觉错觉或混合错误。
  - 证据等级：source-backed（原始 benchmark 的构造和论文报告的模型评估）。
  - 边界：论文所用模型、样本和判定流程不能推出所有 VLM 的错误成因；用作诊断设计参考，不等于证明单个错误的因果根因。

- OWASP GenAI Security Project, [LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)，访问日期 2026-10-29，页面持续更新。实际阅读多模态注入说明与 prevention/mitigation 部分。
  - 来源支持：页面说明注入可来自外部网页和文件，也讨论图像中隐藏指令造成的多模态风险；建议区分/标记外部内容、限制权限、对高风险操作要求人工批准并持续进行对抗测试；页面指出不存在万无一失的防护方法。
  - 证据等级：source-backed（OWASP 当前风险分类和缓解建议）。
  - 边界：这是安全项目的风险指南，不是对某个模型或具体控制措施有效性的独立实验；提示词标记本身不应被写成强制权限边界。

## 统一多模态、音视频推理与交互式世界模型评测

- Xie et al., “MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models,” ICLR 2026；[会议摘要](https://proceedings.iclr.cc/paper_files/papers/2026/hash/d8308c2c45cf95bfebb736be377f0982-Abstract-Conference.html)，[arXiv HTML](https://arxiv.org/html/2504.03641)。实际阅读会议摘要、arXiv HTML 的摘要、§1、§2 和部分 §3 结果。作者把评估分为理解、生成和理解—生成协作三域，使用来自 12 个数据集的任务，另设计五种 unified 子任务；报告的 12 个 U-MLLM 在不同能力维度间存在明显差异。证据等级：ICLR 同行评审 benchmark 论文。边界：结论只适用于作者收录的任务、模型快照、标准化方式和多选题协议；该评测不等于所有多模态任务的统一排名。
- Hong et al., “WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs,” ICLR 2026，[会议摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55dc32df563a35bfebb736be377f098d-Abstract-Conference.html)。实际阅读官方摘要。论文将同步视频、音频与文本输入放入同一评测，作者报告其版本包括 1,662 段音视频、8 个主领域、67 个细分领域、3,172 道选择题和 26 个任务；报告的准确率和标注规模仅代表该 benchmark。证据等级：ICLR 同行评审 benchmark 摘要。边界：摘要不足以复核完整采样、标注一致性和模型提示协议，不能将其结果外推到所有实时音视频系统。
- Rasheed et al., “VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Video,” ICLR 2026，[会议摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e75a1c8af8d9438df1057fdaa42913eb-Abstract-Conference.html)。实际阅读官方摘要。作者设计跨越 10 秒至一小时以上视频的数学任务，重点考察跨时间定位视觉/语音线索、概念迁移和对长解释的理解，并提供多步推理标注。证据等级：ICLR 同行评审 benchmark 摘要。边界：这是数学视频理解的专项测试，不代表一般视频理解或音视频对话能力。
- Ding et al., “PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives,” arXiv:2608.13552v2（2026-08-14），[论文摘要](https://arxiv.org/abs/2608.13552)。实际阅读摘要和版本信息。作者用 Agent Players 针对目标与世界模型交互，避免固定 action sequence 难以跨模型比较的问题；benchmark 包含 171 个场景，并按几何一致性、交互保真、视野外演化、洞察演化及视频质量/可控性评估。证据等级：preprint / source-backed（预印本 benchmark 协议和作者实验）。边界：任务目标、Agent Player、模型快照及自动/人工评估定义限制结论；不能由视频交互分数推出机器人闭环安全或真实控制能力。
- Wang et al., “Vision-Language-Action in Robotics: A Survey of Datasets, Benchmarks, and Data Engines,” arXiv:2604.23001v1（2026-04-24；页面称已被 TMLR 接收），[论文摘要与版本](https://arxiv.org/abs/2604.23001)，[HTML](https://arxiv.org/html/2604.23001)。实际阅读摘要及 §6.1 的部分段落。该综述按数据集、benchmark 与 data engine 梳理机器人 VLA 研究；作者强调真实数据保真度与采集成本的权衡，并指出组合泛化、长时程推理及仿真到真实 grounding 的评估缺口。证据等级：同行评审 survey 对既有文献的归纳。边界：综述观点不是单项受控实验，也不说明这些机器人任务可直接并入通用 VLM benchmark。

以上 benchmark 的数据、输入输出和分数口径彼此不同；不可把它们的准确率或任务结果拼成一张跨 benchmark 排名表。摘要阅读只支持候选任务和协议概览；若正文要引用具体成绩、模型排序或细节，需先阅读相应全文并固定版本。

## 教学公式边界

Patch 与 tubelet 公式给出在固定 patch 大小、边缘补齐和固定 tubelet 约定下的原始空间/时空块数。真实模型可以裁剪、多尺度切块、合并或 resample；原始块数不等于 tokenizer token 数、真实模型计算量或服务成本。该教学关系不构成对某一模型接口的规格声明。

## 生命周期

LLaVA、BLIP-2 和 Flamingo：active 的代表性历史架构，分别用于说明视觉 token 投影、query bottleneck 与 resampler/cross-attention 路线；不代表完整谱系。

Qwen3.5 模型卡及 GPT-4o 产品资料：active 的来源入口，但模型和服务行为须按发布版本复核。

OmniCapBench：2026 年 arXiv 预印本候选；若要引用其完整指标、效果或方法比较，需先通读全文并核对版本。

## 视觉 grounding、时空证据与跨模态评估

- Wang et al., “VGR: Visual Grounded Reasoning,” [arXiv:2506.11991](https://arxiv.org/abs/2506.11991)，当前 HTML 为 v3（2026-05-01）。访问日期 2026-10-29；实际阅读摘要、§3 的视觉回看机制、§5.2–5.3 的对照与消融表。作者描述模型生成区域坐标和回看信号，再从保留的高分辨率视觉特征中检索对应 token，追加到后续推理输入；定位头使用 L1 与 GIoU 目标。证据等级：source-backed（预印本方法和指定实验）。边界：实现基于 LLaVA-NeXT 和其数据/评估协议；不同版本摘要中性能数值有变化，本项目不引用提升数字，不将方法效果外推为所有 VLM 的保证。
- Wu et al., “MagiC: Evaluating Multimodal Cognition Toward Grounded Visual Reasoning,” [arXiv:2507.07297](https://arxiv.org/abs/2507.07297)，2025-07-09 提交。访问日期 2026-10-29；实际阅读摘要、§1、§3.1–3.4。作者报告 5,534 条弱监督样本、896 条人工整理样本和超过 15,000 个推理步骤标注，并分别评估答案正确性、步骤有效性、box grounding 与自我纠错；人工集还含有答案、推理和区域标注。证据等级：source-backed（预印本的 benchmark 构造与作者评估）。边界：数据来自自然图像问答，弱监督数据、人工子集、模型集合和评分定义共同限制结论；它不是生产 grounding 的通用验收门槛。
- Qwen Team, “Qwen3-VL Technical Report,” [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)，arXiv v2（2025-11-27）。访问日期 2026-10-29；实际阅读 HTML 摘要及 §§2.1–2.3。作者描述 interleaved MRoPE、将不同 ViT 层特征注入 LLM 前层的 DeepStack，以及在视频帧组前使用文本时间戳的时间表示。证据等级：source-backed（发布团队报告中的模型设计）。边界：报告描述 Qwen3-VL 的架构与实验，不公开对所有模型普遍成立的内部机制；时间戳表示本身不是精确事件定位的独立证据，也没有在本地复现。
- Wang et al., “XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models,” ICLR 2026，[官方会议摘要](https://proceedings.iclr.cc/paper_files/papers/2026/hash/21071811269c70ac11a3e7869bac747e-Abstract-Conference.html)。访问日期 2026-10-29；实际阅读 ICLR 官方摘要。摘要称 benchmark 包含 60K 道多项选择题、五类任务和六个跨模态方向，并报告其受测模型存在模态差异和方向不平衡。证据等级：同行评审 benchmark 论文的官方摘要。边界：摘要不足以复核完整样本构造和评分细节；结果限定于论文中的模型与协议，不能证明内部表征具有或不具有模态不变性。

## 长时音视频与意图型时间定位

- Tao et al., “LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs,” [arXiv:2603.19217v1](https://arxiv.org/abs/2603.19217v1)，2026-03-19。访问日期 2026-10-29；实际阅读 arXiv HTML 摘要、§1、Table 1、§4.1—4.2。论文构建 275 段 10—90 分钟视频（总计 140 小时，平均约 2,069.7 秒）及 1,014 道人工筛选/标注的选择题；覆盖长时记忆、时间定位、细粒度理解和跨模态感知。Table 3 中 Gemini-3.0-Pro 为 65.8%，Qwen3-Omni 为 35.8%；报告提到本地模型最大帧数、GPU 配置不同，Gemini 采用 1 FPS/deep-thinking 设置。证据等级：source-backed（arXiv v1 benchmark 数据、模型快照和作者实验）。边界：结果绑定选择题协议、所测模型版本、帧采样、API/硬件与推理配置；不能解释为当前产品排名或真实长视频任务成功率。
- Feng et al., “RVG-Bench: A Benchmark for Reasoning-Driven Audio-Visual Temporal Grounding,” [arXiv:2610.32268v1](https://arxiv.org/abs/2610.32268v1)，2026-10-23。访问日期 2026-10-29；实际阅读摘要、§1、§3.1—3.3、§4.1—4.3、结论/限制及附录的人类验证和模态消融。该任务用间接意图查询要求模型结合推理定位目标区间；数据含 1,255 段未裁剪视频和 1,890 组 query-interval 标注（1,588 单步、302 多步），含八种单步子类，超过一半查询需要音频证据。作者评测 22 个模型；Qwen3.8-Omni-Flash 在该协议下得到 49.6 mIoU，人类基线 65.3 mIoU 来自 200 个样本的双人审阅子集。证据等级：source-backed（预印本 benchmark 构造、模型评测和指定人工子集）。边界：视频平均约 108 秒，作者明确未评估小时级电影叙事或长程记忆；模型排名受 API/checkpoint、提示和时间区间评分协议限制，不能外推为全部视频理解或产品能力。

## VLM 多维评估

- Lee et al., “VHELM: A Holistic Evaluation of Vision Language Models,” NeurIPS 2024，[arXiv:2410.07112v2](https://arxiv.org/abs/2410.07112v2)。访问日期 2026-10-30；阅读 v2 摘要。
- 来源支持：作者提出把视觉感知、知识、推理、偏差、公平性、多语言、鲁棒性、毒性和安全纳入 VLM 评估，并标准化部分推理参数、提示方式与评估指标；初始评估覆盖 22 个 VLM 和 21 个数据集。
- 证据等级与边界：source-backed（NeurIPS 2024 论文摘要和指定框架描述）。初始模型/数据集结果是该论文快照与评估协议下的结果，不能作为当前模型排名，也不能替代具体业务的正确性、风险阈值和人工标注。
