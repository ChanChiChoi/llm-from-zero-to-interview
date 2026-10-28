# 第二册整册结构审视与外部扫描（2026-10-28）

## 范围与方法

本记录服务于第二册按新版 `WRITING_SPEC.md` 进行的整册精修。册级阶段先读取简介、目录、书系位置和各章标题/入口摘要；不在骨架决定前逐篇深审全部正文。外部发现扫描聚焦 2025—2026 年能够影响“进阶架构、推理预算、多模态边界”的论文与正式会议论文页，发现候选后回到原始摘要/方法陈述核对任务、作者结论和外推限制。该扫描有明确范围，不代表穷尽 LLM 文献。

## 书册契约

第二册服务于已掌握第一册主线、需要讨论机制、方案取舍、成本账本和实验设计的读者。其独特职责是把“结构和目标怎样形成能力、系统怎样承载训练与推理、证据能支持到哪里”讲成一条进阶路线；第一册负责共同基础，第三册负责可运行项目，第五至第二十四册承接训练、推理、评估、安全、数据、Reasoning、Agent、多模态、架构、Infra 和 Serving 等专题深挖。

因此本册采用进阶教材与研究判断地图的混合类型。它可以设置多条岗位选读路径，但主阅读路径仍应展示依赖和交接，不能退化为并列术语索引。新方向只有在与本册读者任务适配、可构成完整解释链、不会被一个过短讲次“点到为止”时才进入本轮正文。

## 目录与原结构判断

原简介承诺概率、架构、预训练、训练系统、后训练、推理、长上下文、RAG/Agent、Reasoning、论文和多模态，但目录只列十个名词型部分，没有说明前后问题如何接续。简介还称其为“进阶总书”，与第五至第二十四册的专题主讲关系说得不够清楚。

前九部分总体形成可保留的研发链：建模与优化基础 → 架构 → 数据与预训练 → 规模化训练 → 行为塑造 → 推理交付 → 上下文与工具系统 → 推理时计算 → 论文与研究判断。各部分讲数多，不等于必须拆成更多 Markdown 文件；其正文精修需继续检查单篇是否共享同一个上位问题，并在扩充时保持连接。第九部分现有代表论文内容与前面章节部分重叠，章级精修应把共同职责明确为“读论文、检验证据、辨认贡献和边界”，不能重讲同一技术百科。

原第十部分同时容纳 105—124 的多模态概览、图文理解、VLM 对齐、文档与 grounding、图像扩散、自回归图像、视频、Whisper、TTS、实时语音、多模态评估和开放议程。它在任务输入输出、数据、评价指标和系统时序上跨了多个独立主线，单文件 15,753 行难以形成合宜的读者节奏。决定按主题拆成四个文件，讲次编号稳定：105—113 多模态理解与对齐，114—119 图像与视频生成，120—122 语音识别/生成/交互，123—124 评估与开放研究。旧文件保留为指向新目录的兼容说明，不再作为正文入口。

## 外部扫描与候选去向

| 候选 | 已检查的主要来源 | 对第二册的含义 | 决定 |
| --- | --- | --- | --- |
| 离散扩散语言模型与非自回归生成 | [LLaDA](https://arxiv.org/abs/2502.09992)、[LLaDA2.0](https://arxiv.org/abs/2512.15745)、[iLLaDA](https://arxiv.org/abs/2606.25331)、[Continuous Latent Diffusion Language Model](https://arxiv.org/abs/2605.06548) | 这些工作覆盖 masked diffusion 训练、模型扩展/从 AR 模型适配、全双向模型及连续潜空间路线，提示它已形成不止一种实验方案。论文/预印本中的规模、准确率和效率是各自实验结果，不足以推出普遍胜出或生产部署优势。完整讲解需要同时覆盖目标函数、迭代/块级解码、比较口径和 Serving 约束。 | 是重要候选，但当前 124 讲骨架不为新方向匆促增加一讲；记入 `ideas.txt`，第二轮与第二十一册架构主讲位置共同规划章组。第二十四册继续承接真实推理系统和服务接口。 |
| 推理时思考长度的收益边界 | NeurIPS 2025 正式论文 [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e22bea3b170f4c2aebb9c48d98ae64d-Abstract-Conference.html) | 作者在特定数学任务上报告过长 CoT 可能损害部分领域表现，并提出按领域调整长度分布的方案。这支持把预算从“越长越好”改成收益、正确率、成本和任务分布的联合问题。 | 纳入第八部分定向精修和示例；结论绑定论文任务与模型，不外推为所有推理模型的一般规律。 |
| 训练目标与推理搜索的配合 | NeurIPS 2025 正式论文 [Rethinking Fine-Tuning when Scaling Test-Time Compute](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e8f4eae0a41cab67fdead3aa6b77f083-Abstract-Conference.html) | 论文研究 pass@N 下 CE 训练过度自信与搜索表现之间的任务条件关联，提示训练和推理策略并非完全独立。 | 在第 1、5、8 部分的章级扫描中作证据入口，避免写成“交叉熵普遍降低 pass@N”的泛化结论；不新开章。 |
| 全模态统一系统与评估 | ICLR 2026 收录论文入口中出现 OmniBench、WorldSense、AV-Odyssey 等跨语音/视觉评测工作；相关论文需按具体任务与协议逐篇核读。 | 目录已包含跨模态理解、生成、语音和评估。单个 omni 模型的输入输出能力不能取代任务级指标；仍要分开识别、语义/时间对齐、生成、轮次交互和工具行动。 | 不增“Omni 模型名录”章节；在第十至十三部分强调接口和分解评估，具体模型与在线功能由第二十一册模型案例、第二十四册 Serving 和第十五册多模态专题承接。 |
| 具身模型、世界模型与自主驾驶 | 相关会议和期刊方向可作为候选发现信号；详见 `ideas.txt` 中已核查的 NeurIPS/ICLR/ICML workshop 与期刊专题清单。 | 这些主题需要状态预测、动作条件建模、闭环控制、仿真到真实迁移和安全评估，远超本册多模态理解/生成概览的篇幅和任务边界。 | 当前不塞入第十至十三部分；留在 `ideas.txt`，作为 LLM 相邻专题第二轮判断。 |

## 骨架决定

1. 简介重写为“进阶原理与研究判断”，明确目标读者、能力承诺、书系边界和选读路线。
2. 目录把旧有 10 部分的知识清单改写为“目标—结构—数据—系统—行为—交付—应用—求解—研究”的主线，并为每部分说明中心问题、依赖和交接。
3. 保留 1—124 讲编号和前九个文件边界；这些编号已被多个索引使用。单篇是否需拆讲由逐章宏观审视决定，不用编号数量代替结构判断。
4. 将旧多模态巨型文件拆成 4 个主题文件，明确理解/对齐、图像视频生成、语音交互和评估/开放研究的职责。讲次与引用锚点不改。
5. 修正 `BOOK_SERIES.md`、`ROADMAP.md` 和 `README.md` 的 120 讲旧数字与定位描述。
6. 离散扩散语言模型不在当前首轮精修中浅写成名词介绍；以 `ideas.txt` 作为第二轮候选，届时与第二十一册对齐主讲职责并决定是否新增章组。

## 逐章工作边界

骨架调整不表示所有正文都已完成。后续逐章阶段应一次处理一个文件，依次完成章节问题/承接审视、章级外部扫描、内容重排与补足、完整数值例子和可视化、代码状态标记，以及目录/索引同步。旧精修记录只用于复用已核验代码和来源，不能替代本轮对篇幅、连贯性、章节契约和现有正文的实际检查。每章的未完成项和状态追加到 `progress_v2.md`。

## 第七部分：章节结构复读与章级扫描（2026-10-28）

### 章节地图与结构决定

本部分覆盖第 76—86 讲，连续回答“如何选取并使用上下文、如何把证据转换为受约束的行动、如何管理跨任务状态”。复读后保留当前文件，不拆分讲次或重排：

| 子链 | 讲次 | 主要问题 | 交接 |
| --- | --- | --- | --- |
| 长上下文 | 76—77 | 扩展窗口与有效使用如何分别训练、评估 | 进入外部知识检索 |
| RAG 证据链 | 78—81 | 解析、召回、向量表示、重排、claim 支持和引用如何衔接 | 从证据进入外部动作 |
| Agent 执行与状态 | 82—85 | 工具 schema、规划、安全、长期记忆如何保持可控 | 共同用于综合业务流程 |
| 贯通案例 | 86 | 是否能把政策证据、实时状态、验证器和人工确认组成闭环 | 收束并交接到第 87 讲的推理验证 |

这些主题不是并列名词：长上下文提供证据处理容量，RAG 选择外部材料，attribution 检查声明依据，工具和计划改变外部状态，Memory 则管理长期信息；86 讲使用同一退款场景贯通它们。11 讲和 2,900 行左右的单文件仍符合一章的学习节奏，没有证据表明某一段可脱离主问题而独立成册；目录已将本部分定位为“上下文、取证、行动和跨轮状态”。

### 章级外部扫描及去向

| 候选与来源 | 发现 | 决定 |
| --- | --- | --- |
| [LongBench Pro](https://arxiv.org/abs/2601.02872) | 自然产生的双语长文任务和 8K—256K 输入提示，补充合成单 needle 之外的长上下文评估视角；任务规模、语言和评分范围仍有限。 | 纳入 77 讲，作为基准选择提醒；不新增专讲，不把分数当作业务能力证明。 |
| [Relevant Is Not Warranted / FORCEBENCH](https://arxiv.org/abs/2605.28044) | 相关段落可能不能支持声明的数值、时间、范围或强度；论文报告绑定其 198 对样本和四个 judge。 | 纳入 81 讲，将引用存在、相关性和证据 warrant 分开；数字按原评测设置限定。 |
| [Memora](https://aclanthology.org/2026.findings-acl.1337/) 与 [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/) | 长周期记忆需要评估撤回、过期、冲突与后续行动；两个基准的任务契约不同。 | 纳入 85 讲，强调记忆更新和遗忘生命周期；不拼成单一通用分数。 |
| [Agent Memory: Characterization and System Implications](https://arxiv.org/abs/2606.06448) | 记忆构建、检索和生成形成状态化工作负载并各有成本。 | 纳入参考来源，支持生命周期和读写分离；研究样本不代表全部 Agent 系统。 |

本轮没有发现必须新开讲次的章级主题。RAG/Agent runtime、工具协议、跨任务记忆安全与评测的专题深挖交给第二册后续相关册和书系既有章节，保留当前层级的机制地图和交接关系。

### 正文增补与门禁状态

章首新增从上下文、解析/权限、证据、动作到 trace 的流程图。补充或重算了长上下文位置差异（`D_mid=0.23`、位置鲁棒比 `0.733`）、RAG gold evidence 覆盖与 1,300-token 预算、1,000 万向量的 FP16/PQ 内存账本、召回/精排的评分和批处理成本、规划关键路径、长期记忆过期评估，以及退款案例的金额、版本、token、幂等和状态竞态。相应讲次 76—86 当前解释性正文均达到 1,800 个汉字门槛；按口径排除标题、代码、公式、表格和状态标记，逐讲表保存在 `progress_v2.md`。

本轮没有修改可执行代码，原验证标记继续有效；新计算均为文中合成教学输入并已手算。`PAPERS.md` 加入 LongBench Pro、FORCEBENCH、Memora、Mem2ActBench 和 Agent Memory 的研究入口。尚未完成外部 MathJax/GitHub 实际渲染、全章旧链接 HTTP 扫描、真实 RAG/Agent 或 serving 实测，因此第七部分不标为全部门禁完成。

## 第八部分：章节结构复读与章级扫描（2026-10-28）

### 主线和结构判断

第八部分第 87—96 讲构成“怎样产生、扩展、验证、训练和评估推理”的一条机制链：CoT 提供中间状态，多路径采样增加候选覆盖，verifier 做选择，search 决定扩展路径，test-time compute 管预算，数学/代码训练与合成数据形成能力来源，推理评估报告能力剖面，最后第 96 讲将这些部件放入产品控制循环。章首 Mermaid 图表达运行时闭环，95 讲是跨讲指标汇总，96 讲把数据、训练、运行时、产品输出和发布评估串起来。

复读后保留现有单文件及顺序。各讲回答不同但相邻的问题，且可以复用同一组“候选—验证—成本”概念递进；没有发现应把某几讲合并成更大概念篇或拆成独立章节的证据。第九部分继续负责论文阅读方法，第七部分负责 RAG/Agent 通用系统，故第八部分的资料检索与训练数据讨论留在推理任务语境，不扩成第二个 RAG 或数据百科。

### 定向扫描及候选去向

| 论文/方向 | 核对来源和发现 | 正文去向与证据边界 |
| --- | --- | --- |
| [KisMATH](https://aclanthology.org/2026.tacl-1.59/) | TACL 2026 页面摘要介绍因果 CoT 图和图对齐干预，报告 1,671 个数学题、15 个开放权重模型。 | 纳入 87，作为步骤干预研究案例；自动抽图、数学数据和模型集合限制外推，不当作通用 CoT 忠实性证明。 |
| [RAG over Thinking Traces](https://aclanthology.org/2026.emnlp-main.1964/) 与 [ThinkRetrieve](https://aclanthology.org/2026.findings-emnlp.1703/) | EMNLP 2026 页面描述检索轨迹语料、离线结构化与 retrieve-then-generate；ThinkRetrieve 研究按推理步骤取回已求解过程。 | 纳入 92，将两种轨迹检索设置与事实 RAG 区分。所报收益绑定各自模型、任务、语料和处理协议，不推出普遍增益或替代 verifier。 |
| [Learning from Synthetic Data Improves Multi-hop Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e95016eceae77e945b84554b0f1bb49-Abstract-Conference.html) | ICLR 2026 正式会议页描述规则生成、仅含虚构知识的合成数据 RL，并报告在作者 QA 基准中的迁移表现。 | 纳入 94，用于区分“学习事实”与“学习组合操作”；只按论文配方和评测任务呈现。 |
| [Native Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html) | ICLR 2026 页介绍从标准 QA 对生成自身轨迹、把 reasoning 视为潜变量的 NRT 框架。 | 纳入 94 作训练目标候选；不据摘要推断其解决所有不可验证任务，仍需独立看过程质量与策略塌缩。 |
| [FOL-Traces](https://aclanthology.org/2026.findings-eacl.115/) | Findings of EACL 页面介绍程序化核验的一阶逻辑轨迹、masked operation prediction 和 step completion。 | 纳入 95 作过程评估资源；一阶逻辑和作者的模型/任务设置不能代表广泛自然语言推理。 |

这些研究没有触发新增章节：CoT 结构干预、轨迹检索、合成训练和过程评估均可放回现有讲次，且能增强“生成—验证—训练—评估”的接续。平行轨迹和 wall-clock compute 的 PaCoRe、ParaThinker、Timely Machine 作为同一章节的并行预算补充。完整论文/会议页链接已同步 PAPERS。

### 正文和门禁状态

章首流程图把生成器、状态、verifier 和预算控制器连接起来。各讲通过独立算例区分候选覆盖与多数票、verifier 覆盖与条件选择、树搜索节点预算、按任务分桶分配额外计算、轨迹模板迁移、合成样本新颖性，以及答案与过程双指标评估。所有新增数字均标为教学构造；推理长度、搜索、合成训练和轨迹检索的论文结论限定在作者样本与协议中。

第 87—96 讲当前正文长度、代码未变/验证状态、Mermaid 与 MathJax 外部渲染未验风险和本轮具体校验见 `progress_v2.md` 对应第八部分条目。论文信息来自所列官方会议信息页及 ACL Anthology，未声称逐篇复核所有全文附录、代码或复现结果。

## 第九部分：论文精读与开放研究题（2026-10-28）

### 主线和结构决定

保留 97—104 讲的单文件与顺序。链条为：Transformer 的序列交互机制 → GPT/指令模型的目标变化 → 预训练规模与预算 → DPO 偏好目标 → LoRA 参数适配 → FlashAttention 内存访问 → MoE 稀疏容量与调度 → 开放研究设计。第 104 讲把前面各讲带回可证伪研究问题；没有发现需要新增独立讲次或重排的主题。新增解释集中于适用边界、可复算估算和不同指标的区分，不以摘要式论文名录扩展正文。

### 章级扫描与纳入判断

| 方向与来源 | 对现有主线的贡献 | 证据边界与去向 |
| --- | --- | --- |
| [Pre-training under infinite compute](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79682ef54677a2cb3dfb8383b372037f-Abstract-Conference.html)、[Pretraining Scaling Laws for Generative Evaluations](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d7fbeddfd9bb6dace146cc4fa02ff0-Abstract-Conference.html)、[How Text Quality Interventions Reshape Neural Scaling Laws](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html) | 扩展固定数据与重复曝光、生成式 pass@k 评估、数据质量干预对 scaling 拟合的影响。 | 纳入第 99、104 讲及 `PAPERS.md`。作者实验和外推绑定各自模型、数据、指标与计算范围；未改写成新规模定律。 |
| [What Matters in Data for DPO?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3f37b8fbd43303106dd141a602838ad5-Abstract-Conference.html) | 提醒偏好学习的数据审计要检查 chosen 样本质量、负例作用及偏好支持范围。 | 纳入第 100 讲与对齐论文入口；不推广为 rejected 样本普遍无用。 |
| [LoFT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html) | 把低秩参数化与低秩子空间中的 optimizer state 分开考察。 | 纳入第 101 讲和 `PAPERS.md`；内存与质量结果绑定论文模型和目标层，不代表所有 LoRA 实现。 |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) | 作为硬件感知算法与 kernel pipeline 协同的最新案例，强调 baseline 版本和硬件 shape 的重要性。 | 纳入第 102 讲和推理优化资料；预印本结果绑定 Blackwell/B200、BF16、指定 cuDNN/Triton 版本及 workload。 |
| [Multi-Layer Scheduling for MoE-Based LLM Reasoning](https://arxiv.org/abs/2602.21626) | 区分模型 router 均衡、expert 通信和请求级服务调度。 | 纳入第 103 讲和架构/推理索引；作者 TTFT/TPOT 数字绑定其系统实验，不外推为通用收益。 |

### 正文改动、算例与验证

补入 DPO 偏好环和 prompt 内分数不可跨任务校准的解释；补充 LoRA rank 上限、初始化梯度路径及两个投影的完整参数/显存账本；补 online softmax 可合并摘要、FlashAttention tile/硬件约束及 prefill/decode 口径；补 MoE 硬分配与 soft router 概率的区别和系统级诊断；补 scaling residual 与开放研究样本量的手算。当前 97—104 讲解释正文依次为 **1,922、2,095、2,023、1,899、1,891、1,901、1,950、1,912** 个汉字，按同一脚本排除代码、公式、表格、图示、标题和状态标记，均达 1,800 下限。

由于发现原 LoRA 参数账本比较了两个 full projection，却只计入一个 adapter，已将代码修正为两个目标投影的 adapter 参数 $2\times r\times(d_{in}+d_{out})$，同步手算结果为 131,072 个 adapter 参数、0.391%、256 倍参数差异。只抽取并执行这一发生实质变化的代码块，Python 3.12.9 输出 `merge_max_diff=4.440892098500626e-16`、`trainable_percent=0.391`、`reduction_x=256.0`；其余代码、接口和依赖未变化，继续沿用原验证状态，没有重跑。`PAPERS.md` 已同步 scaling、DPO、LoFT、FlashAttention-4 和 MoE 调度入口。

## 第十部分：多模态理解与对齐（2026-10-28）

### 结构判断与近期方向

保留 105—113 顺序：多模态任务地图 → CLIP 表示对齐 → vision encoder → connector → LLaVA/Flamingo/BLIP 架构 → 指令训练 → 偏好对齐 → OCR/图表/文档 → reasoning 与 grounding。该顺序从表示逐步走向任务行为和证据支持，未发现需要增加独立讲次或拆分现有讲次的主题；113 讲篇幅较长，但其 grounding、claim-evidence、空间关系、多图引用和证据干预共同回答“多模态回答凭什么可信”，保留为一条完整主线。

章级扫描核读 ICLR 2026 [XModBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/21071811269c70ac11a3e7869bac747e-Abstract-Conference.html) 摘要。其核心问题是统一音频、视觉和文本模型是否具有跨模态一致推理、是否保留模态特有偏差。纳入 113 的评估边界并同步 `PAPERS.md`；该 benchmark 不足以证明模型内部形成模态不变表示，也不能忽略各模态信息量、输入质量和预处理差异。

### 视觉结构、例子与状态

章首新增 Mermaid 证据流图，说明输入表示、对齐/连接器、语言模型回答、定位结果和证据检查之间的回路。105—113 解释正文计数为 **4,330、2,730、3,039、3,283、3,457、3,207、3,019、3,202、8,255** 汉字，均超过 1,800。原有代码共有 9 个 Python 块、全部有邻近状态标记；本轮没有改变可执行代码、API、依赖或结果承诺，未重跑。XModBench 的结论只绑定其任务和评估协议。

## 第十一部分：图像与视频生成模型（2026-10-28）

### 章节主线与扫描

保留 114—119：生成目标总览 → diffusion 数学 → latent diffusion → 自回归图像 token → 视频时空建模 → 从 open-loop 视频到 world-model 证据。图像和视频共享条件生成问题，但跨帧对象、运动和状态一致性要求增加，因此继续在一个生成分支里递进展开。

章级扫描补读已收录的 ICLR 2026 [Scaling Laws for Diffusion Transformers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/880d8999c07a8efc9bbbeb0c38f50765-Abstract-Conference.html)，正文在第 114 讲区分直接实验点和曲线外推，并说明结果不构成通用 DiT 指数。另核读 2026 [WBench](https://arxiv.org/abs/2605.25874)：289 个案例、1,058 轮交互，以视频质量、场景遵循、交互遵循、一致性和物理合理性组织 22 个子指标。作者发现没有单一模型在所有维度都强；结果受其样本、控制接口、自动指标和模型快照约束。纳入第 119 讲，用来区分画面逼真与交互式世界建模，不另开章节；更完整的具身/驾驶闭环专题仍按书册契约留待书系第二轮安排。

### 图示、例子与状态

章首新增从条件、表示、采样到图像/视频质量评估的 Mermaid 分支图；已有逐步扩散、latent 元素量、VQ 图像 token、视频时空成本、对象持续性和物理一致性算例继续用于机制解释。114—119 讲计数为 **5,507、4,035、4,290、2,667、3,084、3,400** 汉字，均达下限。六个原有 Python 教学块均保留 2026-10-19 验证标记；本轮只增正文、来源和图，不重跑未变化代码。图像生成质量、世界模型和服务效果均不从作者 benchmark 外推到生产产品。

## 第十二部分：语音识别、生成与交互（2026-10-28）

### 主线与近期全模态系统案例

保留 120—122：语音转写与识别误差 → TTS、韵律和说话人生成 → 端到端流式对话、turn-taking 与 barge-in。目录与正文将识别、合成和对话视作不同接口，并以播放状态和用户可听内容作交接。

章级扫描核读 Qwen 团队 [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)。报告描述 Thinker-Talker MoE 处理文本、图像、音频和视频输入并生成文本/流式语音；论文给出冷启动下理论 first-packet latency 234ms。正文将其作为原生全模态交互的具体案例，并明确该数值是论文的理论估算，不是线上 p95 或跨硬件实测。该案例无需新开 Omni 名词讲次；它增强 122 的架构和延迟比较，模型接口细节由 `PAPERS.md` 专题资料承接。

### 图示、例子与状态

新增章首 Mermaid，分出级联 ASR—LLM—TTS 与端到端语音输入输出，并把打断事件闭合到轮次状态。120—122 计数为 **3,672、4,923、5,552** 汉字，均达下限；三个已有标准库 Python 块的状态标记保留。本轮代码未改、未重复运行。发布讨论继续同时考虑语义正确、RTF、首包延迟、打断和安全，不用 Qwen 技术报告中的理论延迟替代目标部署测量。

## 第十三部分：多模态评估与开放研究（2026-10-28）

### 结构判断与章级扫描

保留 123—124 两讲：123 建立能力、grounding、生成、安全和部署评估对象；124 以共同研究骨架把多模态助手、幻觉/证据、文档、生成评估、视频世界模型、实时语音和多模态 Agent 变成研究问题。第 124 讲篇幅较长，但各方向统一按场景、baseline、瓶颈、数据、评估与风险展开，并由实验设计和研究路线两节重新收束；它承担的是研究议程综合，不是独立的多模态名词词典，所以保留在本册收束位置。

章级扫描核读 2026 年预印本 [LVOmniBench](https://arxiv.org/abs/2603.19217)：275 段 10—90 分钟视频、1,014 组 QA，聚焦长程记忆、时间定位和跨模态理解；作者报告的开源模型低于 35%、该设置中的 Gemini 3 Pro 约 65% 绑定论文数据、模型快照及评分协议。正文将其纳入 123 的长视频切片设计，强调短 clip 结果不可代替长时音画同步评估；不把论文数值写成当前产品排行榜。

### 图示、例子与状态

章首新增从场景合同、能力切片、证据标注、指标与区间、发布门槛、线上反馈回到新问题的 Mermaid 闭环。第 123、124 讲解释正文计数为 **4,632、10,822** 汉字；原有两个教学代码块均有 2026-10-19 邻近验证标记，本轮没有改变代码，不重跑。现有数值例子包括多任务评估分母、grounding 与安全风险、研究方案评分和阈值门禁；论文指标均限定在作者实验范围。

## 第二册图示与全书静态复核（2026-10-28）

本轮给第 10—13 部分各新增一张标准 `mermaid` 图；第 1—9 部分原有 Mermaid 保留。目录中的 13 部分文件入口均与现有文件对应，旧 `10-多模态大模型进阶.md` 作为兼容跳转页，不计作第十四个正文单元。对第二册目录、简介及章节内 Markdown 相对链接做静态存在性检查，未发现缺失链接；外链 HTTP 全量扫描和 GitHub 实际 Mermaid/MathJax 渲染仍未完成。

本轮另核对 [GitHub Mermaid 图示文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams) 对 `mermaid` 语言围栏的要求，以及 [Mermaid flowchart 语法文档](https://mermaid.js.org/syntax/flowchart.html) 的连线和文本标签写法。第二册 18 个图块均使用标准反引号围栏；这只确认源码形态与文档约定相符，不能代替把当前未提交版本放到 GitHub 后的页面渲染检查。

对 1—13 份正文按 `WRITING_SPEC.md` 排除代码、公式、表格、图示、标题和状态行后统计：每个现存核心讲解单元均超过 1,800 个汉字。第 1—4 部分最低单元分别为 1,874、1,847、1,904、1,858；第 5—9 部分最低为 1,814、1,805、1,860、1,876、1,891；第 10—13 部分最低为 2,730、2,667、3,672、4,632。Python 状态标记静态检查覆盖 1—13 部分全部既有 Python 块；本轮只执行了发生实质变化的第 101 节 LoRA block，其余块未重复运行。

仍需完成 GitHub 实际渲染、未扫描外链的整体检查，以及从简介/目录沿主路径的连续复读。以上未完成项不抹去本轮章节结构和正文修订，但第二册完整渲染与书册门禁保持开放。

## 第一、二部分：训练稳定性与 Transformer 架构新研究扫描（2026-10-28）

### 检索范围与证据位置

按第一部分训练稳定性、第二部分残差/归一化和 FFN 激活三个问题，核读 ICML 2026 PMLR 正式论文页的 AdaGC、SpanNorm 摘要，以及 arXiv 页面中 PowLU、MoA、Taming Curvature 的题录和摘要。此次仅以官方摘要核对方法目标、实验设置概况和作者主张；没有把摘要级信息扩写成完整方法复现或独立复现实验。相应入口已收入 `PAPERS.md`。

### 纳入与暂缓决定

| 研究 | 对应正文问题 | 决定与证据边界 |
| --- | --- | --- |
| [AdaGC（ICML 2026）](https://proceedings.mlr.press/v306/wang26dg.html) | 全局梯度范数裁剪如何应对不同张量的异常尺度 | 纳入第一部分第 7 讲一段案例，说明按张量维护历史裁剪尺度的另一种设计。模型规模与 spike/准确率结果只按论文列出的实验理解，不说成普遍稳定保证。 |
| [SpanNorm（ICML 2026）](https://proceedings.mlr.press/v306/chao26b.html) | Pre-LN/Post-LN 与深层残差通路的折中 | 纳入第二部分第 13 讲，作为直接残差路径、输出归一化与缩放联合设计的案例；不把作者结论改成新的通用默认配置。 |
| [PowLU](https://arxiv.org/abs/2605.25704) 与 [MoA](https://arxiv.org/abs/2605.26647) | FFN 激活的低精度稳定性与 token 条件表达 | 在第 15 讲并列呈现两种不同研究问题。PowLU 与 MoA 仅按各自预印本摘要和实验模型范围介绍，不把两者当成可直接排名的激活函数。 |
| [Taming Curvature](https://arxiv.org/abs/2606.16768) | 曲率估计与逐步增加网络深度的稳定性 | 记录在 `PAPERS.md` 作为研究入口，本轮暂不写入教学正文。摘要描述的在线 Hessian-vector product 估计和 architecture warm-up 会引入另一条训练调度主线；需先读全文与实现条件，再判断它是否适合第 7 讲的篇幅和边界。 |

### 正文结构审视

第二部分第 13—15 讲分别负责残差递推与深度稳定、norm 统计轴与数值实现、FFN 的表达/参数/激活账本。新增研究仅用于更新各自接口，不另增讲次。第一部分第 7 讲负责训练状态诊断，因此 AdaGC 放在全局裁剪之后；optimizer 算法细节仍由第 9 讲负责。章节改写后按讲重新计算解释正文篇幅，并保留手算例，不以新增论文名替代机制说明。

## 第三部分：Tokenizer、数据与预训练进阶（2026-10-28）

### 主线与结构决定

保留第 26—40 讲的单文件和顺序。该部分回答“什么文本以什么表示进入模型、训练预算怎样分配、预训练结果如何被可信地测量和恢复”：Tokenizer 与工程协议（26—27）→ 数据来源、去重和 mixture（28—30）→ 合成数据与 scaling（31—32）→ loss、稳定性、数值格式、checkpoint 和停止评估（33—37）→ 多语言与代码语料（38—39）→ 可复核训练方案（40）。它比书系第九册的全链路数据治理更聚焦训练目标、token 暴露量和预训练验收；不同于第四部分对集群执行的主讲。

### 章级扫描和候选去向

围绕跨语言 token 成本、词表可变性、混合配比、重复曝光、合成数据质量、文本质量 scaling 和抗污染评估，核对 NeurIPS 2025、EMNLP 2025、ICLR 2026 和 ACL Anthology 的正式论文页或作者预印本页。研究用于补充已有机制接口，没有发现需要新增独立讲次或拆开这条预训练链的证据。

| 候选来源 | 纳入位置与可支持结论 | 证据范围 |
| --- | --- | --- |
| [Explaining and Mitigating Crosslingual Tokenizer Inequities](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b91cefbfa52340af2f16f249572dc76-Abstract-Conference.html) 与 [The Token Tax](https://arxiv.org/abs/2509.05486) | 纳入 26、38；用 tokenizer 的 fertility/token premium 解释固定 token 预算下不同语言可容纳文本量的差异。 | 前者报告 7,000 个单语 tokenizer、97 种语言的研究；后者在 16 种非洲语言和约 9,000 道选择题的范围内研究 token 使用与准确率关系。不能从相关性推出 tokenization 的因果效应，也不能外推到未测语言或其他任务。 |
| [Broken Tokens?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b5c5689fae6fa9a4883e73e511d52c8-Abstract-Conference.html) 与 [zip2zip](https://proceedings.neurips.cc/paper_files/paper/2025/hash/def682d57535b9e778508b4b1a911034-Abstract-Conference.html) | 纳入 27，说明非规范分词鲁棒性与动态推理词表是不同研究路线。 | 指令微调模型表现不能代表 base model；动态词表结果受论文的 uptraining、词表构造和推理协议约束，不代表任意部署可改变 token ID 协议。 |
| [Scaling Laws for Optimal Data Mixtures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html) 与 [Datasets, Documents, and Repetitions](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aba6ec20299931d46ddeefd5ddcb442-Abstract-Conference.html) | 纳入 30，支持把配比、文档独立性和重复曝光作为联动变量设计受控实验。 | 作者拟合和训练设置不形成普遍的固定 mixture；文档级重复不等同于有效独立数据量。 |
| [Demystifying Synthetic Data in LLM Pre-training](https://arxiv.org/abs/2510.01631)、[Scaling Laws Revisited](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html) 与 [How Text Quality Interventions Reshape Neural Scaling Laws](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html) | 纳入 31—32；区分重写文本、教材式生成、数据质量干预和拟合条件，强调合成比例及 scaling 系数随数据类型和预算改变。 | 研究规模、模型、质量操作和评估目标均依原实验，不整理成三分之一等固定生产配方，也不把不同论文的拟合结果横向排名。 |
| [BeyondBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22b4e30a7a660b17f1fb58ab49671e77-Abstract-Conference.html) | 纳入 37，补充算法生成任务作为固定题面评估的抗污染切片。 | 44 类任务和 117 种变化形式由生成规则决定其覆盖面；可降低记忆原题的风险，但不证明没有污染，也不能代表开放知识或真实产品任务。 |

### 本轮正文和局部校正

正文仍由 tokenizer 接口一路推进到预训练决策；在 30 讲重新核算 1T token 预算下各数据桶的实际曝光倍数，并在 31 讲区分筛选通过率、人工抽检精度和总体误收不确定性。32 讲将数据质量干预和 scaling 曲线放在拟合边界内解释；33—37 讲分别承接 loss 分母、训练异常、精度误差、恢复一致性和停止证据；38—39 讲把 token 预算及可执行代码语料放回各自语言/任务范围，避免一次主讲同时覆盖数据治理全学科。

逐讲解释性正文统计为：第 26—40 讲依次 **2,330、2,079、2,128、1,958、1,875、1,876、1,882、1,827、1,838、1,850、1,881、2,007、1,835、1,804、2,275** 个汉字，均超过 1,800 下限。计数排除标题、代码和图示围栏、表格、行内公式/代码、引用列表、状态标记及空白；它只作门禁，不能替代阅读。发现第 37 讲一处行内统计符号丢失反斜杠，已修复，并补入一组 200 题 checkpoint 对照，说明点差异小于抽样不确定性时仍需检查配对题与训练成本。第 40 讲增加到下一部分的因果交接，因此当前正文数高于初次统计。现有代码、接口和依赖未改变，沿用原验证状态。

本部分来源入口已补到 PAPERS.md。公式和链接的静态检查及整册连续阅读仍在进行；此处记录章级研究与当前复核，不据扫描或篇幅统计代替目标数据集实验。

## 第四、五部分：规模化训练系统与后训练更新（2026-10-28）

### 结构判断

第四部分第 41—50 讲继续按“数据并行与梯度语义 → 状态分片 → TP/PP/SP/CP → 混合并行 → 激活与通信调度 → 吞吐测量 → 事故恢复”推进；十讲仍围绕一个问题：如何在保持训练语义的同时把同一训练目标放到多卡执行，并能诊断和恢复。激活卸载归入第 47 讲，因为它与 activation checkpointing 同属“反向时如何拿回前向状态”的资源交换；异步 RL rollout 的队列、长尾和版本新鲜度归入第 49 讲，因为它改变训练吞吐与测量口径。第 50 讲仍承接实际故障和恢复。无需增设独立讲次或改动顺序。

第五部分第 51—65 讲仍按“从 Base Model 到助手行为 → SFT 数据与泛化 → 偏好数据和 reward → PPO/DPO 等目标 → RLVR 与反馈信号 → 安全、真实性和鲁棒性 → 评测与验收”展开。第 56 讲已经负责在线策略优化和多轮决策接口；多轮 Agent 的局部 credit 分配作为 56.11 小节补充它，不另开完整的 Agent RL 讲。训练系统如何产生异步轨迹归第四部分，轨迹内状态/动作怎样形成策略信号归第五部分，环境轨迹的复现与运行验收交第七部分。

### 章级外部扫描与候选去向

| 主题与来源 | 处置 | 可支持内容及边界 |
| --- | --- | --- |
| [TorchTitan graph-based CPU activation offloading 工程记录](https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/) | 并入 47.9，不新增讲次。 | traced graph 选择、view replay、pinned-memory pool、NUMA、预算/defer/prefetch，以及 HBM、重计算、主机内存和传输带宽的交换。其公开结果绑定 8×H100 与文中具体模型、batch、sequence、预算；MoE 的 timeout/开销结果不能泛化成架构定律。 |
| [PyTorch 2.13 distributed release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/) | 并入 50.14，作为监控和通信恢复接口的版本案例。 | `torchcomms` 条目标记 API Unstable，发布说明提到 structured logs、collective tracing、timeout 和 partial-group recovery；不据此推断优化器、数据游标、checkpoint shard 已一致或完整训练自动恢复。 |
| [DORA async RL system](https://arxiv.org/abs/2604.26256) | 系统调度细节并入 49.10；56.10 仅保留算法相关的 policy-version 接口并链接第四部分。 | 多版本 streaming rollout、轨迹派发时固定行为策略版本及 staleness window 是预印本系统方案；最多 2.12×端到端和 8.2× rollout 数值受作者任务、模型、基线和实现约束。 |
| [Relax omni-modal async RL engine](https://arxiv.org/abs/2604.11554) | 作为 49.10 多模态异构 rollout 的系统案例，不独立成章。 | 模态/角色服务隔离与异步传输队列帮助解释多模态 Agent rollout 的系统负载；Qwen3-4B 与 Qwen3-Omni-30B 的公开 speedup、reward 收敛只支持论文对照配置，不能和 DORA 横向排名。 |
| [GiGPO（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf) | 并入 56.11，作为重复环境状态下的 step-level 局部相对信号；不新增讲次。 | 在轨迹级信号外，按多条已采样轨迹中的同状态 anchor 比较局部折扣回报。依赖可比任务/初始状态、可靠 state key 和重复 anchor；论文在 ALFWorld、WebShop、搜索增强 QA 及 Qwen2.5 checkpoint 条件下的结果不能推为通用 Agent 收益。 |
| [SPMD Types in TorchTitan](https://docs.pytorch.org/devlogs/distributed/2026-08-26-spmd-types-in-torchtitan/)、[PyTorch DTensor 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html) 与 [TorchTitan config-based sharding 工程记录](https://docs.pytorch.org/devlogs/distributed/2026-08-17-config-based-sharding-full-dtensor/) | 并入 42.11 的实现更新和分片矩阵算例，不建立独立讲次。 | TorchTitan 记录将该项目中的 DTensor 持久状态与 SPMD Types 计算区域区分，并报告在其 eager 预训练/RL backend 的迁移；这是框架表示/实现演进，不改变模型的全局分片语义，也不构成所有训练栈的 API 默认值或性能保证。 |

### 实际修改、数值例和视觉

- 第 47.9 节并列表达常驻、重计算和 CPU 卸载分别消耗 HBM、FLOPs 与主机/互连带宽。TorchTitan 案例中增加 0.46 GiB 峰值空间伴随 60—70 tokens/s 下降，按端点折算约 130—152 tokens/s/GiB；这是该配置的局部斜率，不是线性定律。MoE defer 结果保留其模型和集群边界。
- 第 42.11 节补入 TorchTitan 中 DTensor 持久分片与 SPMD Types 计算表示的项目级迁移，并用 8-rank DP=2/TP=4、`W` 为 16×64 BF16 的教学例，逐步算出全局权重 1,024 个元素/2,048 bytes、每 rank 常驻 128 个元素/256 bytes、all-gather 后本地 TP 切片 256 个元素/512 bytes；该账只覆盖一块权重张量。
- 第 49.10 节新增同步 rollout 长尾手算：8 个 worker 中 7 个各运行 2 分钟、1 个运行 10 分钟，总工作 24 worker-min、窗口容量 80 worker-min，粗略利用率 30%，已完成 worker 空等共 56 worker-min。随后用 Mermaid 图将策略版本、环境版本、等待队列、陈旧度门控、丢弃/修正/重采样和下一次更新连成闭环；该静态算例只隔离长尾，不把 `1/0.3` 当作真实系统加速。
- 第 50.14 节增加 `torchcomms` 的接口状态和通信恢复与训练状态恢复的界线。
- 第 56.11 节新增 GiGPO 折扣回报算例：同状态下两步后成功、一步后成功和最终失败的回报分别为 0.81、0.90、0；使用教学简化 `F_norm=1`，均值 0.57，局部 advantage 为 0.24、0.33、-0.57。正文同时说明真实组合还包含 episode-level advantage，且 state key 的遗漏会污染分组。
- 全部只改既有章节内的小节与引用，没有增删讲次或移动原有主题；代码语句、接口、依赖和预期输出均未改变，沿用原有验证记录。

### 同步和待办

来源入口和证据边界已补入 `PAPERS.md`；局部信用分配与激活卸载题目已加入 `EXERCISES.md`；术语表补入 Activation Offloading 和 GiGPO。正文 Mermaid 节点使用引号标签，带文字箭头采用管道标签。需继续做第二册全路径连续复读、所有新旧外链的全量可达性检查，并在 GitHub 实际确认 Mermaid/MathJax 渲染；论文系统性能也尚未在目标训练栈、GPU 集群或目标 RL 环境复测。
