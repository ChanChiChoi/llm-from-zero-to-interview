# 第四册《大模型百科全书》架构审视与精修记录

## 书册契约

- **目标读者与前置**：正在学习书系任一主线、需要快速确认大模型术语、公式、方法边界和相近概念的读者；默认具备一定 Python、机器学习或深度学习基础，但不要求先读完本册。
- **读者所得**：能定位核心概念、识别常见方法的适用条件与混淆点，并从本册入口进入有连续讲解、例子和工程边界的专题正文。
- **范围**：基础数学、模型组件、训练、推理、服务、RAG/Agent、评估、安全、多模态、论文索引、方法比较及有来源约束的代表性前沿更新。
- **不承担**：不按课程顺序替代主书，不以条目代替重要机制讲解，不承诺穷尽全部模型发布或持续实时维护排行榜。
- **类型与独特价值**：参考工具书和书系导航层；相邻主书讲机制与实践，本册提供稳定分类、最小定义、边界提示和交叉入口。论文索引提供证据入口，面试比较表提供横向复习。
- **前后交接**：概念主干接书 01–03、05–24 中的专题正文；推理与部署重点指向书 06、23、24；Agent、评估和协议分别指向书 02、07、17、20、22；架构深讲指向书 21。

## 简介审视与调整

- **原问题**：简介“内容结构”只列 16 个主题组，把第 16–18 章合并为一个多模态条目，并遗漏第 19 章；没有逐章说明读者问题、主线作用或后续入口。
- **调整**：简介现按 19 章逐条提供短摘要及章节链接，覆盖基础、训练与系统、应用与评估、证据索引、多模态和前沿更新。摘要突出查询目的和下一步，不复述正文。
- **后续要求**：逐章精修完成时复核对应摘要是否仍准确；若章节定位或边界改变，回到本记录更新章节地图与简介。

## 目录审视与调整

- **原状**：19 个章节文件均列出，且提供 serving、训练、Agent、架构等跨册路径；但章节列表没有稳定分类，读者需从文件标题自行归类。
- **调整**：按基础概念、数据/训练/系统、上下文/推理/治理、证据/比较、多模态、前沿更新六类组织 19 个入口，保留现有跨册阅读路径，明确本册按问题查找而非线性通读。
- **链接状态**：简介和目录共检查 49 个 Markdown 链接，其中 49 个为本地目标，全部存在；逐章链接与外链终审保留在相应章节记录和整册收口门禁。

## 简明章节地图

| 章 | 本章作用 | 依赖与交接 |
| --- | --- | --- |
| 01 数学与机器学习 | 数学、概率统计、实验和 ML 基础术语入口 | 为 02、04、06、12 提供前置 |
| 02 深度学习组件 | 网络组件、损失、优化与训练环路速查 | 接 03/04，详细实现转入实战与训练册 |
| 03 NLP 与 Tokenization | 规范化、预切分、分词、token ID 和文本输入协议 | 接 04/05 与预训练数据主题；完整 tokenizer/数据格式实践由第五册主讲 |
| 04 Transformer 架构 | 注意力、位置、结构变体与长序列概念地图 | 接 05/08/09，深讲转书 21/24 |
| 05 大语言模型基础 | 语言建模、能力、评估和产品基础概念 | 接 06–12 与对应主书 |
| 06 预训练与数据 | 语料、训练阶段、数据治理和 scaling 概念 | 接 07 与书 05 |
| 07 后训练与对齐 | SFT、偏好优化、RL 与可验证反馈概念 | 接 08、书 05 及对齐专题 |
| 08 推理与解码 | 从概率分布到可观测生成行为 | 接 09 与书 24 |
| 09 推理优化与部署 | serving、性能、cache、量化与运行时入口 | 接书 06、23、24 |
| 10 分布式训练系统 | 并行、通信、故障恢复与平台运维入口 | 接书 05、23 |
| 11 长上下文、RAG 与 Agent | 信息组织、检索、工具和 Agent 状态 | 接书 02、07、17、20、22 |
| 12 Reasoning 与评估 | 推理时计算、验证器和评估口径 | 接书 07 及 reasoning/评估专题 |
| 13 安全与治理 | 威胁、隐私、审计和治理框架 | 与 11/12 交叉，接相应安全专题 |
| 14 重要论文索引 | 按主题提供论文与阅读入口 | 横向回链 01–13、16–19 |
| 15 面试高频对比表 | 按机制、条件、代价和失败边界比较方法 | 横向回链 04–13、16–18 |
| 16 多模态学习 | 跨模态表示、对齐、融合与感知任务 | 接 17/18 与多模态专题 |
| 17 图像生成与视频生成 | 视觉生成方法、控制与评估入口 | 接扩散/生成专题与多模态系统 |
| 18 语音大模型 | 语音理解、生成、实时交互和指标入口 | 接 16 与语音专题 |
| 19 前沿架构与模型更新 | 代表性机制和发布方披露的时点快照 | 简介级索引接书 21；服务与运行时细节接相应主书 |

册级阶段依据简介、目录、章名和先前结构盘点形成地图，没有把 19 篇正文当作已完成逐章审阅；章级问题、教学单元、例子、证据和视觉检查在逐章周期补录。

## 册级发现扫描

- **日期与窗口**：2026-10-30；主要关注 2025-10-01 至 2026-10-30 的架构与模型披露。
- **问题**：是否出现足以改变本册分类或要求新增独立章节的架构主线；模型更新章节是否需要与第二十一册现有深讲区分；近期官方模型披露是否揭示可迁移概念缺口。
- **资料类型**：原始预印本摘要、官方模型卡、模型发布说明和官方文档；榜单/论坛不作为架构事实依据。检索不是穷尽证明。
- **候选与去向**：
  - **Multi-Stream Attention Residuals（MSAR）**：2026-10-09 arXiv 预印本将 AttnRes 扩展为多头深度路由，仍属于深度方向残差选择，暂不新开章节；进入第 19 章候选简条，并在书 21 第 75 章审阅时判断是否作为 AttnRes 后续小节。完整论文和实验表尚待阅读。
  - **Gemini 4 Argon**：官方模型卡披露 sparse MoE Transformer 与文本/视觉/音频输入；书 21 已有第 95 章专门处理。第 19 章只补简短索引或回链，不重复深讲。
  - **Mistral Large 4**：官方资料显示多模态 MoE 与开放权重交付，适合作为第 19 章模型快照候选，不构成新的架构类别。官方快照存在 52B 与约 49B active 参数差异，正文需固定权重 revision 并核对对应 config。
  - **GPT-6.1 Terra**：OpenAI 发布说明强调视觉理解、电脑操作、成本和发布方评测，但未披露模型内部架构；仅在第 19 章的选择规则涵盖接口产品快照时考虑，不据此增加架构内容。
  - **Claude Haiku 5.5**：Anthropic 官方索引列有 2026 年 10 月系统卡，但本轮无法读取卡片正文；暂不形成能力或架构结论，章级复核时再决定是否纳入。
  - **新增主线判断**：上述候选分别归属已有的深度残差、多模态 MoE 或发布方模型信息分类；本轮未发现足以支持第四册新增独立章节的证据。继续保持 19 章骨架。

## 架构决定、覆盖与视觉机会

- 保留 01–19 顺序和现有边界，不新增、拆分、合并或迁移章节；本册的概念速查与第二十一册的架构深讲继续分层。
- 第 14 章论文索引与第 15 章方法比较承担横向入口；第 16–18 章分别承担多模态学习、视觉生成和语音主题，避免继续合并为一个“多模态”目录项。
- 视觉机会在逐章阶段按机制检查：04 注意力信息流；07 对齐方法关系；09 serving 数据路径与 KV 生命周期；10 并行拓扑；11 检索/工具状态流；16–18 的跨模态或生成流程；19 的架构轴对照。适合时优先复核现有表图，不为装饰增图。
- 第 19 章使用动态快照，应保留来源日期、模型准确 ID、接口与架构证据的区别、未披露边界，并把深机制链接到主专题；不能变成排行榜穷尽目录。

## 待处理风险与工作交接

- 当前第 19 章链接到 `research/model-update-2026-09/` 下的多份底稿；该目录在本工作区不存在。第 19 章周期须逐条验证并修复断链或补齐有证据的研究记录，不能把缺失底稿误报为完整证据链。
- 第 19 章的近期模型快照需按精确版本复核；Mistral Large 4 参数字段有官方快照差异，Gemini 4 Argon 的书 21 第 95 章已存在，需核对两册入口避免重复。
- 第 15 章已有用户工作区改动，新增“精确 attention 目标不等于跨 kernel bitwise 相等；稀疏 mask 改变全连接配对集合”的说明。该改动保留，并在第 15 章逐章审阅时纳入核验。
- 本记录只完成册级架构审视，不代表任何章节已通过正文门禁。简介/目录链接静态核对已通过；适用图示的目标平台渲染和全书最终门禁仍待完成。
- **下一步**：第 01—03 章已完成章节扫描与静态精修，接续第 04 章《Transformer 架构》；逐章更新对应摘要和本记录。暂不更新 PDF。

## 第 01 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/01-数学与机器学习.md`。本章定位是跨主题术语索引，读者用条目确认定义和常见边界，再沿“深入阅读地图”进入书 13 的数学主讲或书 5 的训练主讲；条目不承担完整课程。逐项检查定义域、易误解的结论、公式方向、指标口径、交叉链接和主讲入口。
- **chapter_scan**：扫描问题为：索引是否漏掉关键的概率/优化/统计评估与对齐概念；哪些条目将局部启发式写成通用保证；公式是否说明分母、方向、取值域和实现约定；哪些重要条目缺少可到达的完整正文。范围截至 2026-10-30，重点核对 DPO 目标、Dropout 行为、ROC-AUC/Precision-Recall 定义、频率学派均值置信区间及 BH/BY 的依赖边界；材料包括原始论文、PyTorch / scikit-learn / SciPy 官方文档及 NIST 手册。扫描不是数学教材定义的穷尽证明。
- **候选处置**：没有发现需新增独立主题或拆章的证据。新增 DPO 显式目标及假设，保留完整算法讲解在书 5；补清 perplexity 的有效 token 聚合约定、MAP 正则化与损失缩放关系、按指标方向解释过拟合/分布偏移差值、Accuracy/ECE 非空定义、置信信号与正确概率的区别、MDP 折扣因子域和 RLHF KL 实现约定。修正互信息/RAG 直觉，说明检索证据可能改变或复杂化答案分布，不保证降低答案不确定性。候选定义均留在速查层，推导沿书 13 与书 5 的现有入口。
- **length_status**：独立核心讲解单元“把公式变成可检验结论”有 1,847 个解释性汉字（移除标题、围栏代码/公式/示意后计数），达到 1,800 下限。其余条目按本章契约属于支持型索引，不是可独立学习的辅助讲解；条目提供定义/使用线索并指向主讲，避免将词典条目扩成重复正文。
- **example_status**：保留有效 token 加权交叉熵的 `6/11≈0.545` 与样本等权平均 `1.2` 对照；LoRA 的 131,072 对 16,777,216 参数、比例 128，并说明不等于整体显存缩减；新增五样本 paired lift 手算（旧版 0.4、新版 0.8、绝对差 0.4 / 40 个百分点、相对变化 100%）及配对、不确定性和失败分母边界。公式一致性段还说明如何复算交叉熵和成功任务成本。
- **evidence**：DPO 公式按 Rafailov et al., arXiv:2305.18290v3 §4 Eq. (7) 核对，假设和 token log-prob 约定紧随公式说明；公式证据同步补入 `research/rlhf-preference-and-policy-optimization-2026.md`。Dropout 训练/评估语义、ROC 与 PR 指标、均值置信区间、BH/BY 条件分别依据 PyTorch 2.12、scikit-learn 1.9.1、NIST 和 SciPy 1.18.0 官方材料；适用范围不外推到其他版本或评测协议。
- **code_status**：一个纯 Python 标准库 `math` 示例。可执行语句未改；新增了依赖和运行方法、预期输出及局部性质边界。当前标记“静态审阅，未运行（2026-10-30）”，手算预期依次为 weak_loss≈0.9398、strong_loss≈0.2294、shift_invariant=0、equal_large_loss≈1.0986。本轮没有执行该代码。
- **formula_status**：补齐交叉熵支持域与 nats 单位、PPL 有效目标 token 口径、DPO display 公式和适用假设；约束空评估集、ECE 空桶和空 batch 的处理。行内/围栏公式定义域和推导后的口径已静态复核。
- **link_status**：全文 26 个 Markdown 链接，本地目标缺失 0；代码、math、text 等围栏 9 对闭合。关键新增 DPO 来源及正文中的 NIST、SciPy 来源已核对；其余外链最终状态仍随本册外链终审复核。无 Mermaid 图。
- **sync_status**：简介中的第 01 章摘要仍准确；目录和主讲入口未变。未发现需要重复同步 PAPERS、EXERCISES、术语表或知识图谱的新增主题；DPO 的既有研究记录已补公式来源。
- **open_risks**：本章示例代码未运行，不能标为执行验证；第四册尚有 02–19 章待扫描/精修，整册连续复读和外链终审未完成，不能据本章结果标记整册完成。

## 第 02 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/02-深度学习组件.md`。以“decoder-only Transformer 的组件如何组成可训练系统”为整章主线，保留 2.1–2.17 顺序；校准它在百科册中的参考入口定位，增加到实战和架构主讲册的导航。
- **chapter_scan**：检查组件主题是否缺少 reader route；公式的 MAC/FLOP、归一化、优化、梯度累积、数据并行和注意力口径是否准确；label shift、mask、checkpoint、DDP、异步计时等是否有版本或状态边界；代码是否给依赖、运行方法、预期结果和当前状态。范围截至 2026-10-30，主要核对 PyTorch 2.12 的 BatchNorm、AdamW、AMP、DataLoader、DDP、activation checkpoint、CUDA 计时/内存和 profiler 文档，并对照相关原始论文。扫描未发现需要增设独立主题；loss 与 labels 由本册第 01 章和第五册第 05 章主讲，已在阅读地图增加入口。
- **章节定位与长度**：将 2.1–2.16 作为按 token→组件→更新→数据/分布式→验证/诊断排列的支持型速查入口，而非 16 篇独立教程；各入口已有完整主讲链接。合并阅读边界至第 2.16 节及本章回顾这一整合单元计 10,083 个解释性汉字，超过核心下限。单入口汉字静态计数如下；它们不承担独立教学，依本册百科契约以简明说明加主讲入口处理。2.17 为参考资料索引，不计讲解篇幅。

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 2.1 线性层与 MLP | 889 | 支持型速查 |
| 2.2 激活函数 | 798 | 支持型速查 |
| 2.3 归一化 | 771 | 支持型速查 |
| 2.4 残差与 Pre-LN | 651 | 支持型速查 |
| 2.5 优化器 | 817 | 支持型速查 |
| 2.6 初始化、精度与显存 | 970 | 支持型速查 |
| 2.7 Tensor 语义 | 503 | 支持型速查 |
| 2.8 Autograd | 572 | 支持型速查 |
| 2.9 Module 与 checkpoint | 398 | 支持型速查 |
| 2.10 数据集与 batch | 472 | 支持型速查 |
| 2.11 Training loop | 454 | 支持型速查 |
| 2.12 分布式训练 | 642 | 支持型速查 |
| 2.13 计时与 profiling | 391 | 支持型速查 |
| 2.14 Transformer 组件 | 498 | 支持型速查 |
| 2.15 组件验证 | 367 | 支持型速查 |
| 2.16 故障诊断 | 444 | 支持型速查 |

- **example_status**：补入可复算例子：2.1 两层 FFN 参数 128、768 MACs/1,536 FLOPs；2.2 SiLU(-1)；2.3 LayerNorm/RMSNorm 对 `[1,2,5]` 的输出；2.4 残差导数与确定性方差；2.5 AdamW 首步；2.6 7B 参数 BF16/FP32 与 Adam 状态账本；2.8 链式法则；2.10 shift 后有效 label；2.11 不等长 micro-batch token 均值；2.12 DDP rank 平均与全局 token 均值；2.14 两 key attention；2.15 未来信息反事实检查。例子均为教学构造，不代表硬件或模型实测。
- **evidence**：PyTorch 2.12 的版本化行为及边界记入 `research/pytorch-training-api-boundaries-2026.md`；Embedding、LayerNorm、SDPA 和 mask 的接口事实沿用 `research/transformer-component-apis-2026.md`。官方资料确认 BatchNorm 前向/运行方差口径、AMP API、IterableDataset worker 分片、DDP `no_sync` 和默认梯度平均、checkpoint RNG 状态、CUDA 同步计时、allocator 可见范围以及 profiler tracing 开销。相关原论文仍在第 2.17 节提供读者入口。
- **code_status**：全文 8 个 Python 块均有状态、依赖/版本、运行方式和静态预期结果，统一为“静态审阅，未运行（2026-10-30）”；本轮只改写 §2.4 残差方差示例的可执行语句，将随机增量换成可手算的确定性同向残差。其他块可执行语句未变但缺少可追溯的历史逐块验证，故不沿用“已验证”。本轮未执行代码。
- **formula_status**：修正两层 FFN 的计算量口径：`2BTd_model d_ff` 是 MAC 数，按乘加各计一个 FLOP 则为 `4BTd_model d_ff`；显式区分样本均值与有效 token 均值；补上 DDP 平均梯度时 `W/N_global` 缩放；明确 `no_sync` forward 范围、CUDA 计时同步和 API 内外的 label shift 责任。归一化、AdamW 和 attention mask 的版本边界已贴近结论说明。
- **link_status**：46 个 Markdown 链接，本地目标 16 个且全部存在；导航把线性/激活、归一化、初始化/优化、loss/labels、Tensor/Autograd/Module/Data、training/checkpoint、DDP/profiling、Transformer/cache 入口分流至书 5、14、21 和本册第 01 章。56 个 fenced blocks 的 112 个围栏标记配对闭合，8 个 Python 块各有状态标记；无 Mermaid。外部链接仍须在第四册最终链接门禁全量复核。
- **sync_status**：简介第 02 章摘要仍准确；目录结构未变。PAPERS、术语表等已有主讲入口，无需复制新增内容；新 research 记录只保存 PyTorch 版本化事实和来源。
- **open_risks**：8 个代码示例均未运行；30 个外链中只有本轮新增/使用的关键 PyTorch 文档完成定向阅读，其他外链留待全册终审。第四册第 03–19 章与整册连续复读仍待完成。

## 第 03 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/03-nlp与tokenization.md`。按第四册百科定位，将本章限定为语言模型文本表示与输入输出协议的速查和导航，不承诺覆盖分类、序列标注、句法分析等完整 NLP 任务谱系。3.1—3.11 保留为按数据流排列的支持型入口，完整 tokenizer/数据格式教学链接到第五册第 03 章，预训练与长上下文入口链接到第二册第 03 章。
- **chapter_scan**：检查字符串到模型 ID 的链路是否区分 normalization、pre-tokenization、子词模型、ID 映射与 post-processing；BPE、WordPiece、Unigram 和 SentencePiece 的训练/推理机制是否混淆；特殊 token、模板、label mask、Unicode、多语言效率、窗口预算及 PPL 是否有足够条件和可追溯来源。范围截至 2026-10-30；定向阅读 Hugging Face Tokenizers 当前管线文档、Transformers tokenizer API 当前文档、Google Research BERT WordPiece 源码、SentencePiece 原论文及 BPE/Unigram 形式化与近期方法论文。未发现改变第四册 19 章骨架或新增独立章节的证据；发现的主要缺口是 pre-tokenizer 层级和明确的完整主讲入口。
- **候选处置**：不扩为独立 NLP 任务分类章，以正文范围说明消除标题可能带来的过宽预期；把“十二个独立主题”校正为 3.1—3.11 的百科支持入口，并新增按查询主题指向第五册第 03 章、第二册第 03 章及本册第 01、06 章的主讲地图。新增 tokenizer 管线及一个 pre-tokenization 示例。补入 Transformers 当前主线文档中的 `sanitize_control_tokens` 行为，并限定它只代表该 API 当前文档记录的接口，不外推到旧版本或其他序列化系统。BERT 贪心最长匹配和整词退回 `[UNK]` 明确为 Google 源码这一实现的行为。
- **length_status**：将“阅读边界”、3.1—3.11 与“本章回顾”视作一条从字符串到模型输入再回到文本的整合速查单元，解释性汉字共 **7,432**，超过核心单元 1,800 下限；3.12 是资料索引。单条目计数如下，均标为支持型速查而非独立教程；完整主讲位置见章首导航表。

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 3.1 Token 接口与 token 账本 | 734 | 支持型速查 |
| 3.2 Vocabulary 与 Embedding | 488 | 支持型速查 |
| 3.3 BPE | 1,040 | 支持型速查 |
| 3.4 WordPiece、Unigram 与 SentencePiece | 682 | 支持型速查 |
| 3.5 字节、字符、子词与多语言公平 | 847 | 支持型速查 |
| 3.6 Special token | 623 | 支持型速查 |
| 3.7 Chat template | 516 | 支持型速查 |
| 3.8 Prompt 与上下文窗口 | 502 | 支持型速查 |
| 3.9 Detokenization | 291 | 支持型速查 |
| 3.10 Tokenizer 训练与迁移 | 590 | 支持型速查 |
| 3.11 训练目标与评估 | 554 | 支持型速查 |

- **example_status**：补清 padding 的手算账本：有效 token 18、padded slots 27、waste `1−18/27=1/3`、平均长度 6；BPE 教学语料 `abab` 与 `aba` 中 `ab` 计数 3、`ba` 计数 2，合并后为 `[ab,ab]` 与 `[ab,a]`。保留 Unigram 两条路径概率 0.4 与 0.09、边缘概率 0.49；Unicode 码点/字节/NFC 结果；上下文预算 `600+200+100=900`；相同总 NLL 下不同 token PPL 与相同 BPB 的对照。所有数字是教学输入，不代表模型或语料实测。
- **evidence**：管线职责、预切分输出和 post-processor 边界来源见 `research/tokenizer-pipeline-and-control-token-apis-2026.md` 中 Hugging Face 官方文档记录；当前 Transformers 文档的控制 token 清理参数及可变文档边界亦记于该文件。BERT WordPiece 贪心和未知词回退仅按仓库源码限定。SentencePiece、BPE 形式化和 Unigram 预印本分别记录来源类型及未验证范围。正文按版本与实现收窄结论，没有据单一 tokenizer 推断普遍语言效率或模型质量。
- **code_status**：5 个 Python 块均标为“静态审阅，未运行（2026-10-30）”，声明 Python 3 标准库、保存为 `demo.py` 后运行的方式和手算预期；本轮把 BPE 教学输入改为无并列频率的微型语料，其他代码语句未改。代码没有执行。
- **formula_status**：复核 token/padded 计数和 waste 分母、Unigram 路径概率与边缘概率、有效预测 token 的 NLL 分母、PPL 与 BPB 的单位；上下文预算注明三部分互不重叠并区分窗口、API 输出上限和计费口径。未发现需要改写的公式错误。
- **link_status**：29 个 Markdown 链接，本地目标 10 个且缺失 0；新增管线文档、控制 token API、BPE 形式化与近期 Unigram 预印本入口。82 个围栏标记（41 对）配对闭合，5 个 Python 块均有状态标记；无 Mermaid。其余外链待第四册最终链接门禁。
- **sync_status**：简介第 03 章摘要已改为规范化、预切分、子词算法、控制 token、模板和评测口径；目录结构和标题不变。`EXERCISES.md` 已有 tokenizer 覆盖与计数练习入口，无需重复同步题目。`research/tokenizer-pipeline-and-control-token-apis-2026.md` 仅记技术事实与来源，不含本章工作日志。
- **open_risks**：5 个示例代码均未运行；Transformers `main_classes`、Tokenizers `latest` 页面是滚动文档，正文只作当前 API 边界说明，后续若绑定具体发布版本须按该版本复核。其余外链仍待整册终审；第四册第 04—19 章和整册连续复读未完成。

## 第 04 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/04-transformer架构.md`。按本册参考工具书定位，把 4.1—4.13 校准为顺着 Transformer 信息流排列的百科支持入口；章首新增按查询主题进入第一册、第二十一册和第二十四册主讲的导航。章节保留原有结构和主题范围，不替代专题书中的完整机制推导。
- **chapter_scan**：检查 token 到 logits 的数据流、位置编码与 RoPE 外推、attention 数例、mask 和 cached decode 的绝对位置、MHA/MQA/GQA/MLA、KV 生命周期与显存、full/local/sparse/linear attention、FlashAttention 的精确性边界、Transformer block、encoder/decoder/Cross-Attention、长上下文利用、递归状态和 NoPE、MoE 负载与总/激活参数，以及验证入口。范围截至 2026-10-30；定向核对 PyTorch 2.12 SDPA 文档、FlashAttention 代表论文、Gated DeltaNet 与 KDA 论文、Gated DeltaNet-2 arXiv v1、Kimi Linear 和 DeepSeek-V2。未发现需要改变第四册骨架或为本章新增独立主题的证据。
- **候选处置**：FlashAttention-1 至 -4 的实现演进作为 4.7 支持入口的补充，不把单一硬件结果外推为通用加速；Gated DeltaNet-2 作为 4.11 中递归状态的近期研究补充，链接第二十一册第 62 章，不新增独立百科章；PyTorch 2.12 非方形 `is_causal=True` 左上对齐写入缓存 query 的边界说明和显式绝对位置 mask 示例。Gated DeltaNet-2 的退化关系按论文收窄为“erase/write 同一标量时回到 KDA；channel-wise decay 也退化为标量时回到 Gated DeltaNet”。
- **length_status**：将 4.1—4.13 与本章回顾作为一个连贯架构速查单元，解释性汉字共 **6,737**，超过 1,800 核心单元下限。逐入口排除标题、代码/公式围栏、表格、状态标记和纯参考资料链接后，4.1—4.13 均为支持型速查，完整主讲位置见章首导航表；本章回顾计入整体，不单列为知识单元。

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 4.1 Transformer 解决了什么问题 | 516 | 支持型速查 |
| 4.2 Token Embedding 与位置表示 | 860 | 支持型速查 |
| 4.3 Self-Attention 与信息路由 | 544 | 支持型速查 |
| 4.4 Mask 与信息边界 | 679 | 支持型速查 |
| 4.5 MHA、MQA、GQA 与 MLA | 384 | 支持型速查 |
| 4.6 KV Cache、Prefill 与 Decode | 482 | 支持型速查 |
| 4.7 Full、Local、Sparse、Linear 与 FlashAttention | 733 | 支持型速查 |
| 4.8 Transformer Block | 346 | 支持型速查 |
| 4.9 Transformer 形态与 Cross-Attention | 402 | 支持型速查 |
| 4.10 长上下文与有效利用 | 331 | 支持型速查 |
| 4.11 递归状态、Delta 变体与 NoPE | 763 | 支持型速查 |
| 4.12 Mixture-of-Experts | 278 | 支持型速查 |
| 4.13 复杂度、验证与资料边界 | 230 | 支持型速查 |
| 本章回顾 | 189 | 整合收束，不单列 |

- **example_status**：补齐可复算例子：RoPE 二维旋转后点积 `-1`；attention scores `[1,0,-1]` 的 softmax 权重约 `[0.6652,0.2447,0.0900]` 及 value 输出 `7.0125`；长度 8 的 full/local 因果配对数 36/21；`L=2,B=1,T=4,d_h=2,b=2` 下 MHA/GQA/MQA cache 为 256/128/64 字节；`L=32,B=4,T=8192,H_kv=8,d_h=128,b=2` 的 KV cache 为 4.0 GiB；标量 delta state 从 2 到 2.5、再到 2.75；MoE 教学例的总/激活参数为 6.1B/4.1B。以上均为假设输入，不代表模型或硬件实测。
- **evidence**：PyTorch 2.12 SDPA 的非方形 causal bias 左上对齐、布尔 mask 方向和 `attn_mask`/`is_causal` 互斥，按官方文档固定版本记录于 `research/transformer-component-apis-2026.md`。Gated DeltaNet-2 的门控定义、退化关系和 recurrent/hybrid 实验边界依据 arXiv v1 原文，并已有 `research/transformer-architecture-research-2026.md` 证据记录；FlashAttention 的版本演进和硬件限制依据各代论文；MLA/KDA 机制分别链接 DeepSeek-V2 与 Kimi Linear 原文。正文把论文方法、框架 API、教学构造和硬件性能作为不同证据层。
- **code_status**：5 个 Python 块（causal mask、PyTorch SDPA/cache 对齐、KV 显存估算、标量 delta 更新、MHA/GQA shape）均标为“静态审阅，未运行（2026-10-30）”，提供依赖、运行方式和手算预期。本轮只改正文和入口，不改 Python 可执行语句；未运行书中代码、测试、模型或 benchmark。
- **formula_status**：静态复核 RoPE 相对位移点积、scaled softmax、KV 元素/字节换算、因果 full/local 配对数、delta state 方向和 MoE active-parameter 账本；变量、输入形状、单位及算例步骤在相邻正文给出。PyTorch cache mask 明确按绝对 query/key 位置建立。
- **visual_status**：保留章首 token-to-logit 数据流、KV/形状账本和复杂度对照；这些表示足以读清本章作为导航层所需的主路径与成本轴，本轮未新增 Mermaid 图。
- **link_status**：章节 46 个 Markdown 链接（26 个本地、20 个外链），26 个本地目标均存在；简介 19 个本地链接均存在。88 个围栏标记（44 对）配对闭合，5 个 Python 块都有当前状态。修复章末练习链接的失效锚点，指向现有 Transformer 验收标题。`git diff --check` 通过；20 个外链未做全册终审。
- **sync_status**：简介第 04 章摘要更新为“百科速查入口并进入专题主讲”；目录结构未变。章首导航新增第一册第 02 章、第二十一册相关主讲及第二十四册第 09、35 章直达入口。`PAPERS.md` 已收录 Gated DeltaNet-2 与 FlashAttention-4，`GLOSSARY_EN_ZH.md` 已有 FlashAttention 和 Gated DeltaNet-2，`EXERCISES.md` 已有 attention/缓存练习，不重复粘贴配套内容。研究记录只补版本化接口知识，不记工作日志。
- **open_risks**：5 个代码示例均未运行；20 个外链仅对本轮依据的关键原始论文和 PyTorch 文档完成定向阅读，剩余链接留待第四册终审。第四册第 05—19 章、整册连续复读、最终外链门禁仍未完成，因此不标记本册完成；没有更新 PDF。

## 第四册整书连续复读（第 04 章，2026-10-30）

- **scope**：依主阅读路径完整复读《Transformer 架构》，与第 03 章 token 输入协议及后续模型基础入口对照，检查 token-to-logit 数据流、位置表示、attention/mask、cache、attention 变体、block 组合、长上下文、递归状态和 MoE。
- **continuity_status**：章节边界与主讲导航一致。发现原复杂度段落未明确标记单头口径，补充单头 `O(T²d_h)` / `O(T²)`，多头计算 `O(H T²d_h)=O(T²d)`、显式 score 存储 `O(H T²)`，并保留融合 kernel 的存储边界。无代码改动，代码状态不变。
- **open_risks**：本章 20 个外链尚未纳入第四册全量终审；连续复读目前完成第 01—04 章。下一步第 05 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 05 章，2026-10-30）

- **scope**：完整复读《大语言模型基础》，按“概率目标—训练后行为—上下文与生成—证据与产品”路径检查各入口和章节回顾，并与第 03 章 tokenizer/template、第 04 章架构和 mask/shift 入口交叉核对。
- **continuity_status**：next-token、label shift、生成/停止、事实性评估、权限与运营闭环之间的定义和边界一致；章节给出有效分母、样本与配置限制，无需本轮改正文。
- **open_risks**：连续复读完成第 01—05 章；第四册外链 579 个仍待全量可达性与证据适配复核。下一步第 06 章，整册门禁未完成。

## 第四册整书连续复读（第 06 章，2026-10-30）

- **scope**：完整复读《预训练与数据》，按来源与许可、清洗风险、去重/污染、数据配比、packing、合成、血缘治理、训练预算、恢复与能力消融检查整条流水线，并与第 05 章 token/loss 口径衔接。
- **continuity_status**：有效目标 token、tokenizer/packing 计数、样本边界、训练与评估污染、checkpoint 数据游标的定义相互一致；明确区分独立样本隔离目标和允许前文延续的流式训练目标，未发现需改正文的阻断项。
- **open_risks**：连续复读完成第 01—06 章；579 个外链的全量终审仍待处理。下一步第 07 章，第四册门禁未完成。

## 第四册整书连续复读（第 07 章，2026-10-30）

- **scope**：完整复读《后训练与对齐》，沿 base/SFT、LoRA、能力回归、偏好与奖励、RLHF/PPO/DPO 家族、RLAIF/安全数据、RLVR/推理轨迹和发布评估检查逻辑与主讲分流。
- **continuity_status**：SFT 的有效目标和 shift/mask 与第 06 章数据/packing 口径一致；各偏好优化变体明确区分数据来源、response 长度归一化、reference、reward 和实现边界；指标分母与样例结果相符，未发现需改正文的阻断项。
- **open_risks**：连续复读完成第 01—07 章；579 个外链仍待全量可达性与证据适配终审。下一步第 08 章，整册门禁未完成。

## 第四册整书连续复读（第 08 章，2026-10-30）

- **scope**：完整复读《推理与解码》，沿模型概率、采样/搜索、停止与预算、候选选择、置信信号、KV cache、延迟指标、speculative decoding 和结构化约束检查讲解顺序与定义域。
- **continuity_status**：解码分布与模型参数、候选覆盖与选择器、请求级 TPOT 与事件级 ITL、目标/草稿分布及终止边界均有区分；与第 07 章训练后评估和第 09 章部署衔接，无需本轮改正文。
- **open_risks**：连续复读完成第 01—08 章；外链全量终审仍待处理。下一步第 09 章，整册门禁未完成。

## 第四册整书连续复读（第 09 章，2026-10-30）

- **scope**：完整复读《推理优化与部署》，沿请求生命周期、工作负载、调度、KV、kernel、runtime、并行、P/D 分离、网络、路由、限流、API、发布和 Agent serving 检查端到端资源状态与章节分工。
- **continuity_status**：与第 08 章的解码、停止和延迟指标衔接一致。发现生命周期示例在 `cleanup()` 中先清零 KV block、再验证状态转换；已改为合法迁移到 `CLEANED` 后才释放计数，并静态复核该代码块。未运行代码。
- **open_risks**：连续复读完成第 01—09 章；579 个外链仍待全量可达性与证据适配复核。下一步第 10 章，整册门禁未完成。

## 第 05 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/05-大语言模型基础.md`。按第四册百科定位，将 5.1—5.14 明确为围绕“next-token 模型如何成为可验证系统”的支持型入口，并在章首增加按问题进入训练、评估、后训练、推理、安全、产品和运行时主讲位置的地图；不替代相邻书册的完整教学。
- **chapter_scan**：检查 next-token 概率、Teacher Forcing、label shift、assistant-only mask、Cross-Entropy/PPL、Base/Instruct/Chat、Prompt/ICL、生成与 CoT、记忆/泛化/污染、scaling、幻觉、产品场景、权限/成本/SLO、评估/反馈/回滚之间是否形成一条解释链；检查短条目是否有主讲入口，代表性结论是否有可复算例、条件和证据。扫描窗口截至 2026-10-30，近期研究重点覆盖 2025-10 至 2026-10；定向核对的来源类型包括原始论文、ICLR 2026 论文页面、Nature 官方摘要/文章索引、Transformers v5.17.0 文档与 OpenTelemetry GenAI 语义约定，并复读 `research/language-model-memorization-2026.md`、`research/hallucination-evaluation-and-safety-2026.md`、`research/evaluation-design-uncertainty-and-contamination-2026.md`、`research/test-time-compute-and-evaluation-2026.md`、`research/scaling-laws-data-mixtures-2026.md`、`research/instruction-tuning-data-and-templates-2026.md`、`research/project-tooling-and-observability-2026.md`、`research/agent-trace-diagnosis-and-online-evaluation-2026.md` 与 `research/tokenization-and-pretraining-evaluation-2026.md`。扫描不是模型能力或评测方法的穷尽证明。
- **候选处置**：未发现需要新增独立章或改变第四册骨架的主题。补入测试时计算的预算分配分类及其预印本边界；加入指定污染训练设置下短程 GRPO 令多种检测信号变弱的 ICLR 2026 结果；补充数据质量 scaling 拟合的任务依赖性；收窄 memorization 论文的 extractability 定义；解释闭式准确率与拒答激励时限定论文的评估设定；补足 OpenTelemetry trace 内容可能含敏感信息的配置提醒。候选仍留在原有百科主题中，主讲去向见章首地图。
- **length_status**：将 5.1—5.14 与本章回顾作为一个从概率目标到可验证系统的整合单元，计 **7,366 个解释性汉字**，高于核心单元 1,800 下限。按第 3.3 节排除标题、围栏内容、表格、行内代码/公式、链接目标及状态标记后计数。5.1—5.14 均为有主讲链接的支持型百科入口，不作为独立辅助教学单元；逐入口计数如下：

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 5.1 语言模型与序列概率 | 515 | 支持型速查 |
| 5.2 Next-token、Teacher Forcing 与 Label Shift | 615 | 支持型速查 |
| 5.3 Logits、Cross-Entropy 与 PPL | 459 | 支持型速查 |
| 5.4 Base、Instruct、Chat 与 Assistant | 652 | 支持型速查 |
| 5.5 Prompt 与 Instruction Following | 411 | 支持型速查 |
| 5.6 In-Context Learning | 543 | 支持型速查 |
| 5.7 Generation 与采样 | 612 | 支持型速查 |
| 5.8 CoT 与可验证计算 | 439 | 支持型速查 |
| 5.9 Memorization、泛化与污染 | 559 | 支持型速查 |
| 5.10 Emergent Ability 与 Scaling | 726 | 支持型速查 |
| 5.11 幻觉与事实依据 | 331 | 支持型速查 |
| 5.12 模型能力与产品场景 | 339 | 支持型速查 |
| 5.13 权限、成本与可运营性 | 552 | 支持型速查 |
| 5.14 评估、反馈与版本迁移 | 471 | 支持型速查 |
| 本章回顾 | 142 | 整合收束，不单列 |

- **example_status**：补入序列概率连乘 `0.5×0.8×0.25=0.1`；masked NLL 两个有效目标的手算 `(-ln0.8-ln0.5)/2≈0.458145`；PPL 的 `0.01→100` 与均匀二分类 `0.5→2`；同一温度采样输入的 softmax 及 seed 首次阈值；企业制度检索与后端权限的分工；4 个相关 ICL 示例中的冲突比例 `1/4=0.25`；税额 `100×10%=10`；100 道题中 8 个字符串重叠只作为核查样本；正确概率 0.4 时不同评分规则下猜测的期望分数 0.4 与 -0.2；逐任务合取率与边际率乘积的区别。以上均注明为教学构造或诊断口径，不代表模型或线上实测。
- **evidence**：`Quantifying Memorization Across Neural Language Models` 按 arXiv v3 限定为给训练文本前缀后用 greedy continuation 测量 extractability，不称作实际请求泄露频率；新增的具体定义、实验设置和边界记入 `research/language-model-memorization-2026.md`。ICLR 2026 污染检测结论限于论文构造的推理模型污染设置；质量 scaling 结论限于论文任务与拟合变量。Nature 论文只用于说明特定闭式准确率评估可能奖励猜测，正文不声称所有训练或线上模型必然增加幻觉；其来源边界见 `research/hallucination-evaluation-and-safety-2026.md`。Chat Template 链接固定至 Transformers v5.17.0；OpenTelemetry 语义约定只支持 trace 字段可能含敏感内容及需配置过滤/权限，不代表某产品默认行为。
- **code_status**：2 个 Python 标准库示例均在代码围栏紧邻处标为“静态审阅，未运行（2026-10-30）”，补齐各自的依赖、运行命令与预期输出。可执行语句未改、未执行；masked NLL 预期为 0.458145，采样例子的手算分布约为 `[0.7710,0.1848,0.0443]` 且 seed 7 首次阈值落在第一个累计区间。预期值不记作运行结果。
- **formula_status**：静态复核序列概率乘积、有效目标分母、负对数和 PPL 换算、温度缩放、ICL 冲突率、评分规则期望值及任务合取率；行内变量与教学输入在正文就地说明。没有执行示例代码。
- **link_status**：章节 61 个 Markdown 链接，其中本地 40 个均存在、外链 21 个；练习链接 `EXERCISES.md#大语言模型基础目标与评估分母` 的标题和锚点均匹配。80 个围栏标记（40 对）配对闭合，2 个 Python 块均有紧邻状态标记，无 Mermaid。`git diff --check` 通过；全部外链的最终门禁留到第四册收口。
- **sync_status**：`简介.md` 第 05 章摘要已更新为概率目标、训练后行为、上下文生成和产品评估导航；目录结构不变。`PAPERS.md` 已包含 Nature 评分激励等相关入口，`EXERCISES.md` 已有目标与评估分母练习，相关术语资料已有覆盖，无需重复追加索引。
- **open_risks**：2 个示例仍未运行；21 个外链仅对本章涉及的关键来源做了定向阅读，仍需第四册外链终审。第四册第 06—19 章、整册连续复读和最终门禁尚未完成；没有更新 PDF。

## 第 06 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 chapters/06-预训练与数据.md。按第四册百科与书系导航定位，将 6.1—6.13 校准为覆盖预训练数据生命周期的支持型入口：目标与指标、来源、质量、去重、隐私/版权风险、混合采样、packing、合成数据、评估污染、lineage、compute、checkpoint 和有效训练预算。章首补充通往第五册、第二册、第七册、第八册、第九册及第二十一册的主讲地图；本章提供速查、最小推演和边界，不替代专题书的完整教学。
- **chapter_scan**：检查数据从采集与许可、质量和风险筛查、去重/切分、配比采样、tokenize/packing、训练和检查点保存到评估污染审计的链路是否完整；复核 NLL/PPL 口径、数据质量证据、dedup 风险、PII 分类、EOS 与 shifted labels、合成数据外推、样本污染判断、数据曝光分母、训练算力公式、恢复状态及有效目标预算。范围截至 2026-10-30；重点核对 ICLR 2026 数据质量 scaling、EMNLP 2025 合成数据、模型崩溃 v2、ACL 2024 packing/masking 与 dedup、GPT-3 污染审计和 Task Contamination；官方边界来源包括 NIST Privacy Framework、NIST SP 800-122、美国版权局 AI 页面与 PyTorch checkpoint 文档。扫描为定向检查，不是语料治理或法律义务的穷尽审查。
- **候选处置**：未发现需要新增独立章节或改变第四册 19 章骨架的主题。增加 6.1—6.13 的生命周期入口图；将数据质量 scaling 和合成数据结果限于论文数据集、模型规模、预算和任务；把 dedup 对记忆/评测影响表述为风险而非普遍因果保证；污染重叠只触发人工复核，不据字符串相等直接认定记忆或泄露；PII 与版权内容转为数据风险分类和来源审查边界，不提供法律结论；packing 按容量解释，不写成必然固定长度；合成数据 collapse 结论限定在所引用实验；checkpoint 的精确恢复依赖状态清单与实现/环境条件。主讲内容留在书 05、07、08、09、21 的对应章节。
- **length_status**：6.1—6.13 与本章回顾构成一条从目标口径、数据治理到预算和恢复的整合速查单元，解释性汉字共 **7,866**，高于核心单元 1,800 下限。6.14 为参考资料入口。逐入口单独承担支持/导航作用，均链接完整主讲位置，不以独立辅助教程验收：

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 6.1 预训练目标与评估口径 | 424 | 支持型速查 |
| 6.2 语料来源与数据契约 | 556 | 支持型速查 |
| 6.3 数据质量与筛选 | 712 | 支持型速查 |
| 6.4 去重、污染与记忆 | 811 | 支持型速查 |
| 6.5 隐私、PII 与版权风险 | 538 | 支持型速查 |
| 6.6 数据配比与采样 | 628 | 支持型速查 |
| 6.7 Packing、EOS 与 loss mask | 953 | 支持型速查 |
| 6.8 合成数据与模型崩溃 | 809 | 支持型速查 |
| 6.9 评估污染审计 | 331 | 支持型速查 |
| 6.10 数据 lineage 与曝光量 | 584 | 支持型速查 |
| 6.11 Scaling 与训练预算 | 503 | 支持型速查 |
| 6.12 Checkpoint 与恢复 | 349 | 支持型速查 |
| 6.13 有效训练目标预算 | 506 | 支持型速查 |
| 本章回顾 | 162 | 整合收束，不单列 |

- **example_status**：新增或校正可复算输入：两个 target probability 0.5/0.25 对应 mean NLL 约 1.0397 nats/token、PPL 约 2.828；质量记录示例的长度与阈值筛选；精确重叠计数作为 review flag；50 条代码样本中 2 条与 5 条医疗样本中 1 条 PII 风险的 4%/20% 分层比率（合并 3/55≈5.45%）；两条序列 padding 到 6 的 5/12≈41.7% 浪费；packing 输出及 3 个有效 shifted targets；合成记录筛选 [True, False]；lineage 9M/12M=75%；N=1B、D=20B 时 6ND≈1.2e20 FLOPs，翻倍 N/减半 D 的估算相同但不保证 loss 相同；第 4,000 步 checkpoint 状态；1M 槽位、100 步、有效率 80%/95% 时 80M/95M 有效目标。数值均为明确的教学构造，不代表语料或模型实测。
- **evidence**：ICLR 2026 数据质量结论按 QualityPajama 任务与所测 100M—8B 模型/训练设定限定；EMNLP 2025 结果只支持论文中混合天然网页与改写数据的具体预算观察；Gerstgrasser 等关于 model collapse 的版本固定为 arXiv v2，并限于其替换/累积生成数据实验和理论边界。ACL packing/masking、dedup、GPT-3 污染审计及 Task Contamination 用于支持对应机制和风险，不将局部研究外推为普遍行业结果。NIST Privacy Framework 为自愿风险管理工具，SP 800-122 的机构范围在正文注明；美国版权局页面只支持生成式 AI 训练报告的发布日期/预出版状态，不作为法律结论；PyTorch 文档仅说明其公开 checkpoint API。新事实和来源证据补入相应 research 记录，审阅和取舍留在本文件。
- **code_status**：5 个 Python 块均标记“静态审阅，未运行（2026-10-30）”，并说明依赖、运行方式和预期输出。本轮 6.7 的 pack_sequences 可执行语句有修改，增加空输入序列和内嵌 EOS 拒绝条件；依用户当前约束未执行书中代码，因此该块保持未运行状态。其余块亦未运行；没有运行模型、benchmark 或自动化测试。
- **formula_status**：静态复核有效目标上的 NLL/PPL、重复与风险率分母、padding waste、配比采样、shifted labels 的跨样本边界、6ND 计算量估算、数据曝光比例和有效目标预算。公式变量、单位、教学假设与论文实测条件均在正文区分；N/D 等估算不替代损失曲线或数据质量结论。
- **visual_status**：新增生命周期导航表，并保留数据处理状态流和 lineage/预算账本。对百科导航层而言，表格加逐步状态文字足以显示本章关系，本轮未新增 Mermaid。
- **link_status**：66 个 Markdown 链接：30 个本地目标且全部存在、36 个外链；88 个围栏标记配对闭合，5 个 Python 块均有紧邻状态。EXERCISES.md 中“预训练百科：数据口径与边界”标题与正文锚点匹配。章节外链仅对关键新增来源定向阅读，整册外链门禁仍未关闭；静态空白检查通过。
- **sync_status**：简介第 06 章摘要已更新为来源/许可、质量/风险、去重、采样、packing、lineage、预算与恢复，并引导书 05/07/08/09。目录未变；PAPERS.md 新增《Is Model Collapse Inevitable?》入口；练习题已有预训练数据口径锚点，无需重复新增。research/scaling-laws-data-mixtures-2026.md 和 research/pretraining-data-packing-checkpoints-2026.md 补入新知识及来源边界，没有写入章节工作日志。
- **open_risks**：pack_sequences 的改动尚无执行验证，五个示例均保持静态/未运行；36 个外链尚未完成册级全量终审；第四册第 07—19 章、整册连续复读和最终链接门禁未完成，因此不标记全册完成。未更新 PDF。

## 第 07 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/07-后训练与对齐.md`，按百科定位将本章组织为从行为塑形、偏好数据和代理奖励，到 RLHF/DPO、可验证反馈、推理轨迹与发布评估的查询入口。章首补充专题主讲地图，将完整训练教学分别链接到第五册第 11、15—17 章、第三册第 04 章和第八册第 02、04、06 章；没有新增或调整第四册 19 章骨架。
- **chapter_scan**：截至 2026-10-30，围绕 2025-10 至 2026-10 的资料窗检查 SFT 目标/mask、PEFT、能力回归、偏好标注与候选分布、RM 校准、PPO/KL、reward hacking、DPO 变体、AI/宪法式反馈、安全指标、RLVR/GRPO 及变体、reasoning effort、on-policy distillation、工具轨迹和发布闭环。来源类型包括原始论文与正式会议论文、arXiv 预印本、厂商技术报告/模型卡、固定版本框架资料，并复核现有 `research/rlhf-preference-and-policy-optimization-2026.md` 等资料。定向扫描不声称穷尽后训练配方或产品行为。
- **候选处置**：将 on/off-policy 偏好候选的模型/阶段差异放入 7.5；将 DQO 的组语义多样性目标放入 7.8；将 spurious reward 与错误标注/验证器噪声分别放入 7.12，明确它们是不同噪声定义；把 DeepSeek V3.2 披露的稳定化措施和工具上下文边界放在 DeepSeek-R1 之后的 7.12。均并入既有主线，不另开章节；其余候选主讲位置见章首导航表。
- **length_status**：7.1—7.14 合计 8,578 个解释性汉字，逐节如下。7.10 是独立辅助比较单元，达到 1,000 字下限；7.12 是核心机制单元，达到 1,800 字下限。其余条目为百科支持入口，明确链接完整主讲位置，不作为独立教程单元。计数排除标题、表格、代码、公式、链接目标与状态标记。

| 单元 | 汉字 | 类型与门槛 |
| --- | ---: | --- |
| 7.1 Post-training | 235 | 主讲导航支持入口，不单列 |
| 7.2 Instruction tuning 与 SFT | 344 | 主讲导航支持入口，不单列 |
| 7.3 Full FT、LoRA 与 QLoRA | 296 | 主讲导航支持入口，不单列 |
| 7.4 能力回归与遗忘 | 187 | 主讲导航支持入口，不单列 |
| 7.5 偏好数据 | 525 | 主讲导航支持入口，不单列 |
| 7.6 Reward Model | 307 | 主讲导航支持入口，不单列 |
| 7.7 RLHF、PPO 与 KL | 271 | 主讲导航支持入口，不单列 |
| 7.8 Reward hacking 与 Goodhart | 294 | 主讲导航支持入口，不单列 |
| 7.9 DPO | 273 | 主讲导航支持入口，不单列 |
| 7.10 IPO、KTO、ORPO 与 SimPO | 1,416 | 辅助单元；≥1,000，通过 |
| 7.11 RLAIF、安全数据 | 822 | 主讲导航支持入口，不单列 |
| 7.12 RLVR、GRPO 及变体 | 1,986 | 核心单元；≥1,800，通过 |
| 7.13 Reasoning effort、蒸馏与工具轨迹 | 944 | 主讲导航支持入口，不单列 |
| 7.14 风险与评估闭环 | 678 | 主讲导航支持入口，不单列 |

- **example_status**：静态复核偏好间隔与策略比率、KL、KTO/ORPO/SimPO 目标、GRPO 组优势、DAPO 过滤、GSPO 序列比率、Dr. GRPO 长度分母及工具轨迹状态变化等算例。典型复算包括 IPO 在 `τ=0.5` 时目标间隔为 1、两侧平方误差均为 0.04；KTO 单样本损失约 0.268941/0.731059；ORPO odds-ratio loss 约 0.223144；SimPO loss 约 0.513015；DAPO 保留混合奖励组后优势为 `[-1,-1,1,1]`；token ratio 2 和 0.5 的 GSPO 几何平均为 1。7.14 样本账本的主要预期输出已列在正文并标为手算，不记作程序实测。
- **evidence**：ICLR 2026 on/off-policy 数据研究涵盖五种模型和 DPO/SLiC-HF，偏好由 PairRM 代理标注、AlpacaEval 2.0 使用 GPT-4-1106-preview 评估；阶段差异为作者假设，不能推广为通用日程。ICLR 2026 DQO 通过回答 embedding 的核矩阵行列式表示组语义多样性；正文区分单次 reward 与 10 条回答上的多样性指标，结果绑定 embedding、kernel、采样和任务。PMLR 正式发表的 Spurious Rewards 与 arXiv v2 正文（§§1–5、相关图表与附录分析）共同支持其 Qwen2.5-Math/GRPO 发现；记录随机 reward 21.4 个百分点与 ground-truth 29.1 个百分点的设置差异、clipping ablation 和既有 code-reasoning 行为，并保留模型依赖边界。Noisy Data arXiv v2 记录 20 条样本中 8 条被误标为错、以及作者在指定数学和 BIRD/Text2SQL 设置中的 clean/noisy 差异；该来源是预印本，且与随机奖励研究不是同一种噪声。DeepSeek-V3.2 技术报告固定为 v1，模型卡 `main` 于 2026-10-30 访问；Speciale 不支持 tool calling 的结论仅适用于该变体。新知识、来源版本、阅读范围与证据边界记入 `research/rlhf-preference-and-policy-optimization-2026.md`。
- **code_status**：仅 7.14 有 1 个 Python 3 标准库示例，代码可执行语句本轮未改；正文说明保存/运行方式、手算预期值和“静态审阅，未运行（2026-10-30）”状态。未运行书中程序或自动化测试。
- **formula_status**：静态核对 Bradley–Terry pairwise 目标、PPO clipped surrogate 与 KL、DPO margin、IPO/KTO/ORPO/SimPO 各自序列概率口径、GRPO 与 DAPO 的组/有效 token 权重、GSPO 长度归一化 ratio、Dr. GRPO 固定预算分母、on-policy distillation CE/KL 关系及比例/风险率分母。逐式变量、假设、response mask 与训练器差异有正文说明。
- **visual_status**：新增跨册主讲导航表；保留工具轨迹状态序列、偏好方法对照及评估账本。当前关系可由导航表、步骤列表与算例表达；无 Mermaid，本轮未新增图。
- **link_status**：45 个 Markdown 链接，其中 12 个本地目标全部存在、33 个外链；130 个围栏标记（65 对）配对闭合，含 1 个 Python、35 个 math、29 个 text，无 Mermaid。新增论文来源直达 ICLR/PMLR/arXiv；模型卡链接用于变体能力边界。外链只完成新增关键来源的定向阅读，未完成册级全量核验。
- **sync_status**：简介第 07 章摘要同步加入 SFT、偏好数据、PPO/DPO、RLVR、安全评估与跨册入口；目录结构未变。`PAPERS.md` 增加三篇新论文且后训练清单编号连续；`EXERCISES.md` 增加 on/off-policy 候选对照、DQO Gram determinant 手算、真实标签噪声与随机奖励区分题；`GLOSSARY_EN_ZH.md` 增加 DQO 条目。未发现需要把简介摘要重复复制到配套材料的情况。
- **open_risks**：33 个外链仍待第四册最终门禁逐项处理；本章 Python 示例保持静态未运行；第四册第 08—19 章、整册连续复读和最终链接/渲染门禁尚未完成，暂不标记全册完成。未更新 PDF。

## 第 08 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/08-推理与解码.md`，维持“模型概率 → 解码行为 → KV/运行时 → 评估账本”的顺序。按百科册契约，本章提供跨主题定义、统一抽象、短算例和排查入口；完整算法实现与 serving 设计沿章首导航分别进入第六册、第七册、第十六册、第二十四册和第十三册，不重复这些主讲正文。
- **chapter_scan**：截至 2026-10-30，定向检查自回归分解、temperature 与概率截断、top-h/epsilon/eta、重复控制、EOS/stop/预算、GNMT 长度项、best-of-N 与 self-consistency、熵与 logprob、KV 字节账本、TTFT/ITL/TPOT、speculative decoding 分布保持、约束解码和解码实验口径。核验资料包括原始论文、固定版 Transformers API/源码、OpenAI API 文档与 vLLM v0.32.0 benchmark 文档；扫描聚焦本章结论，不宣称穷尽所有框架的采样接口或服务实现。
- **候选处置**：Transformers v5.20.0 `top_h`、epsilon cutoff、eta cutoff 收入 8.4 的候选过滤器说明，明确 `top_h` 约束重归一化候选熵、epsilon 使用绝对概率阈值、eta 结合绝对和熵相关阈值；它们是版本化接口，不设为跨框架标准，也不新增独立讲次。补充实现级 repetition penalty 的 prompt 范围与正/负 logit 变换；把 stop 字符串返回规则限制在 OpenAI Completions API；beam 长度项绑定 GNMT §7 Eq. (14)，并说明 coverage penalty 在本节省略。用 vLLM v0.32.0 的定义区分客户端 TTFT、事件级 ITL 与请求级 TPOT；补入 static ensemble verification 改变目标输出分布的边界。未发现需要调整百科册边界或 19 章骨架的候选。
- **length_status**：按正文教学字数口径，移除标题、表格、代码/公式围栏、链接、行内代码与状态标记后统计解释性汉字。8.1—8.14 均作为百科支持入口，不承担独立完整课程，并各自链接主讲位置；8.15 为参考资料，不计讲解单元。

| 单元 | 汉字 | 类型与下限处理 | 完整主讲入口 |
| --- | ---: | --- | --- |
| 8.1 请求契约与生命周期 | 391 | 支持型入口，不单列 | 第二十四册第 3 章 |
| 8.2 自回归与 next-token 概率 | 344 | 支持型入口，不单列 | 第六册第 4 章、第二十四册第 8 章 |
| 8.3 Greedy、sampling 与 temperature | 476 | 支持型入口，不单列 | 第六册第 4 章、第二十四册第 8 章、第十六册第 6 章 |
| 8.4 候选过滤 | 745 | 支持型入口，不单列 | 第六册第 4 章、第二十四册第 8 章 |
| 8.5 Logits 处理器与重复控制 | 475 | 支持型入口，不单列 | 第六册第 4 章、第二十四册第 8 章 |
| 8.6 EOS、stop 与输出预算 | 418 | 支持型入口，不单列 | 第六册第 4 章、第二十四册第 8 章 |
| 8.7 Beam search | 369 | 支持型入口，不单列 | 第六册第 4 章、第二十四册第 8 章、第十六册第 6 章 |
| 8.8 Best-of-N 与 self-consistency | 508 | 支持型入口，不单列 | 第十六册第 3、7 章 |
| 8.9 Logprob、entropy 与不确定性 | 269 | 支持型入口，不单列 | 第七册第 1 章、第十三册第 7 章 |
| 8.10 KV Cache | 400 | 支持型入口，不单列 | 第二十四册第 9、17 章 |
| 8.11 延迟指标与压测 | 602 | 支持型入口，不单列 | 第二十四册第 5、15 章 |
| 8.12 Speculative decoding | 756 | 支持型入口，不单列 | 第二十四册第 31 章 |
| 8.13 结构化与约束解码 | 310 | 支持型入口，不单列 | 第二十四册第 30 章 |
| 8.14 解码实验 | 210 | 支持型入口，不单列 | 第七册第 12 章 |

- **example_status**：静态复算了 logits `[1.2,0.4,-0.1,2.0]` 的 softmax（`[0.253,0.114,0.069,0.564]`，greedy id 3）；temperature 示例的尖/平分布和 `[1e308,-1e308]` 稳定缩放；top-p 前缀与 `[0.7,0.2,0.1]` 的 entropy/top-h 对照（`H≈0.802` nat、`hH≈0.401` nat）；presence/frequency 输出 `[1.5,1.5,-0.2]`；有效 token 预算 `[800,264,0]`；GNMT 长度归一化分数 `-0.456/-0.368`；多数答案 `42`、agreement `0.6`；熵 `0.375/1.099` 与平均 logprob `-2.181`；MHA/GQA KV bytes `4,294,967,296 / 1,073,741,824`、比例 `0.25`；请求级 TPOT/事件级 ITL 两组例子；speculative 接受列表 `[True,False,True,True]` 及残差恢复 target；约束 mask 与配置账本示例。上述均为教学输入和手算预期，不是运行结果、模型实测或服务 benchmark。
- **evidence**：Transformers v5.20.0 GenerationConfig 文档支持 top-h、epsilon/eta 与 generation 参数描述；同 tag 的 `logits_process.py` 支持 repetition penalty 对已出现 token 至多处理一次、decoder-only 默认包含 prompt 及正/负 logit 处理。固定版 assisted decoding 文档支持 assistant-model、prompt lookup、self-speculation、MTP、UAD 和 static mixture 的接口/分布边界。OpenAI Advanced Usage 支持 presence/frequency 公式；OpenAI Completions create reference 支持其 stop sequence 返回规则。vLLM v0.32.0 benchmark 文档支持 TTFT、事件间 ITL 和 `(E2E−TTFT)/(output tokens−1)` 的 TPOT 口径。GNMT arXiv v2 §7 Eq. (14) 支持长度归一化公式，且论文还含 coverage penalty。新增事实、版本、阅读范围和外推限制更新于 `research/decoding-sampling-and-test-time-compute-2026.md`。
- **code_status**：13 个 Python 块分别位于 8.2—8.14，覆盖 stable softmax、temperature/sample、top-p、presence/frequency、token budget、length score、majority、entropy/logprob、KV bytes、latency report、speculative acceptance/residual、logit mask 与 config ledger。全部标记“静态审阅，未运行（2026-10-30）”，均声明 Python 3 标准库和预期输出；8.11 延迟代码给出常规与多 token event 两组输入，8.12 用固定均匀阈值展示接受/拒绝分支。未运行书中代码、测试、模型、benchmark 或 PDF。
- **formula_status**：修正/统一 `\mathrm{softmax}`、`\mathrm{TopK}` 和 `\arg\max` 表达；正文定义概率、熵、长度项、预算、缓存维度与时间单位。全文未发现 `\operatorname`，数学围栏 24 个、配对闭合；未转义美元符号为 0。
- **visual_status**：新增按小节到完整主讲位置的导航表；请求生命周期、token 流和计算流程由文本轨迹及时间线表达。未新增 Mermaid 图。
- **link_status**：全章 17 个本地 Markdown 链接目标均存在；44 个围栏块（24 math、13 Python、7 text）共 88 个标记，配对闭合。13 个 Python 块均在围栏后首个非空行处有状态标记；代码、状态与说明没有 `progress/` 内部路径。新增/使用的关键外部来源已按固定版本定向核验，第四册外链全量终审仍待册级收口；`git diff --check` 通过。
- **sync_status**：`简介.md` 第 08 章摘要仍准确，`目录.md` 结构未变。`PAPERS.md` 增加 GNMT 长度归一化入口；`EXERCISES.md` 增加 top-p/top-h 熵手算及多 token event 的 ITL/TPOT 分母题；`GLOSSARY_EN_ZH.md` 增加 Top-h 与 Epsilon/Eta Cutoff。其他已有面试/术语入口无需重复新增。
- **open_risks**：本章代码未运行；正文外链仍需第四册收口时逐项终审。第四册第 09—19 章、全册连续复读和最终外链门禁未完成，因此不标记第四册完成。未更新 PDF。

## 第 09 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 chapters/09-推理优化与部署.md，保留本册系统工程概念地图定位。新增逐主题深入阅读入口，分别连接书 06 与书 24 的调度、KV、量化、PD、协议、生产运维和 Agent serving 主讲章；未调整第四册 19 章骨架。
- **chapter_scan**：覆盖请求生命周期和阶段账本、工作负载/延迟、continuous batching、KV Cache/PagedAttention/Prefix Cache、FlashAttention/vAttention、量化、runtime 选型、TP/PP/EP/offload、Prefill/Decode、分层 KV、路由与扩缩容、限流/重试/deadline、SSE/异步边界、发布探针/观测、benchmark 诊断和 Agent Serving。对本章内的接口和运行时状态进行定向来源核对，不声称穷尽所有 runtime 或生产配置。
- **candidate_disposition**：把 TGI 从未区分维护状态的 runtime 列表改为带官方维护公告与访问日的状态说明；区分 SSE 标准帧字段和模型 API 的应用层事件；区分镜像拉取、startup、liveness 与 readiness；用独立延迟账本表达 PD 分离与远端 KV fetch/recompute 的得失。没有证据支持把 PD、量化、并行或某一 runtime 写成普遍最优路线。
- **length_status**：9.1—9.18 分别为 478、615、479、428、432、432、311、637、354、460、524、386、299、409、471、447、179、459 个解释性汉字。它们均是第四册契约中的支持型查询入口，带完整主讲正文链接，不作为独立教程篇幅验收；9.19 为参考资料，不计教学单元。
- **example_status**：静态复核请求终态清理、输入输出 token 汇总、有界 prefill 调度、MHA/GQA KV bytes、block table/ref count、online softmax、虚拟页 map、对称 INT4 scale/MSE、runtime 选择器、PD 净收益、四维副本数与 nearest-rank 分位数。手算预期包括 4,096/1,024 MiB 与 0.25 KV 比例、softmax normalizer 1.509952、INT4 MSE 0.002105、PD 35−8−22=5 ms、远端 KV fetch 11 ms 对重算 6/20 ms、160 ms deadline 中剩余 40 ms 小于预计 50 ms prefill。整书复读发现 cleanup() 在状态校验前清零 KV blocks，已调整为先完成合法 CLEANED 转换再归零，避免非法调用失败时破坏活动请求资源计数。新增系统场景均明确为教学假设，不是实测。
- **evidence**：官方 runtime 文档和 TGI 维护提示、TensorRT-LLM 量化说明、NCCL collectives、DistServe/Mooncake 论文、WHATWG SSE、Kubernetes probes、OpenTelemetry traces/context propagation 的来源版本与证据限制加入 research/inference-serving-metrics-and-disaggregation-2026.md。具体 runtime 能力仍需绑定 release/硬件，论文实验受其模型、拓扑、trace 和 SLO 限定，SSE 不规定模型 API 语义，探针文档不覆盖镜像下载事件。
- **code_status**：12 个 Python 3 标准库块均有代码状态、保存/运行方式和手算预期输出，全部标记“静态审阅，未运行（2026-10-30）”。整书复读仅调整 cleanup() 的状态检查与资源归零顺序，并静态复核该块；没有执行书中代码、测试、模型、benchmark 或 PDF。
- **formula_status**：静态复核阶段耗时口径、TTFT/TPOT 分母、KV 字节与压力、online softmax、量化、collective 延迟分解、PD 净变化、fetch/recompute 成本、deadline 分配、replica 安全容量和 nearest-rank。将原 PD 玩具账本改为毫秒收益/传输对比，并把运维约束设为独立布尔条件，避免把非时间复杂度直接加到毫秒。0 个 operatorname，0 个未转义美元符号。
- **visual_status**：新增 18 行主题导航表；请求、发布、stream 和工具路径以状态轨迹/时间账本解释。没有 Mermaid，本轮不新增图。
- **link_status**：全章 98 个 Markdown 链接，其中 58 个本地目标全部存在，唯一 heading anchor 通过静态核对；40 个外链。112 个围栏标记（56 对）闭合，包含 21 个 math、12 个 Python、23 个 text 块；12 个 Python 状态标记均紧邻其代码块。没有 progress/ 内部路径；git diff --check 通过。新增官方来源仅完成本章定向核验，第四册全量外链终审仍待最终门禁。
- **sync_status**：简介第 09 章摘要已更新，目录不变；EXERCISES.md 增加 deadline admission 与远端 KV fetch/recompute 练习。PAPERS.md 已含 DistServe/Mooncake，GLOSSARY_EN_ZH.md 已有 SSE、启动门禁、KV 分离等术语，本轮无需重复添加。
- **open_risks**：示例代码均未运行；40 个外链未完成第四册册级全量终审。第四册第 10—19 章、整册连续复读和最终门禁仍待完成，不标记全册完成。未更新 PDF。

## 第 10 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 chapters/10-分布式训练系统.md，保留 10.1—10.22 的百科系统工程地图定位和章节顺序；在章首增加逐主题主讲导航，连接第五册训练机制与第二十三册 AI Infra 系统章节。没有调整第四册 19 章骨架。
- **chapter_scan**：覆盖训练账本、DP/TP/PP、ZeRO/FSDP、activation 与精度、吞吐/MFU、故障诊断、集群拓扑与网络、数据/checkpoint、容器与 Kubernetes、队列和多租户、TrainingJob、实验追踪、可观测性、SLO、成本、安全、审计和平台架构。定向核验 PyTorch 2.15、Kubernetes device plugin、Kueue v0.20、OpenTelemetry、NIST AI RMF、PaLM 与 DRF 来源；不声称穷尽所有框架、scheduler 或集群实现。
- **candidate_disposition**：无新增独立教程主题；所有 10.1—10.22 均作为支持型查询入口保留，并经导航表链接完整主讲位置。补清 DDP collective 顺序、FSDP1/FSDP2 分片与 state-dict API 差异、DCP 拓扑重分片但不自动恢复应用状态、device plugin 与 gang/placement 的边界、Kueue admission 与 Kubernetes scheduler 职责、OpenTelemetry event 表示和 NIST AI RMF 自愿性质。MFU 计时口径固定为同一统计窗口的 T_wall；流水 micro-batch 扩大时注明样本数和 DP 度不变的假设；DRF 份额算例不冒充完整 DRF 策略。
- **length_status**：10.1—10.22 分别为 890、953、652、674、793、714、748、797、202、285、717、424、850、398、314、247、777、355、484、293、301、349 个解释性汉字。所有条目均按第四册契约作为链接完整主讲正文的支持型查询入口，不按独立教程篇幅验收；10.23 为参考资料，10.24 为小结。
- **example_status**：静态手算覆盖 global batch 2×8×4=64、world size 2×2×2=8、8-rank ring 发送 2×7/8×8=14 GiB、理想带宽下约 14 ms/25 GB/s 下约 0.6 s、TP 切片 shape (4096,1376) 与 512 hidden shard、4-stage/12-micro-batch bubble 3/15=0.2、240 GiB/8 rank 长期状态 30 GiB 及示意峰值 56 GiB、梯度均值/不完整窗口分母、MFU 约 17%、checkpoint manifest 布尔提交门禁、dominant share max(4/16,32/256)=0.25、错误预算 0.01 下 2% bad ratio 的 burn rate 约 2。教学数字按构造输入解释，不是硬件或生产实测。
- **evidence**：PyTorch 2.15 DDP/FSDP1/FSDP2/DCP/pipeline/data 文档、Kubernetes device plugin 文档、Kueue v0.20 admission 与 overview、OpenTelemetry Signals/Logs/Tracing 文档、NIST AI RMF、PaLM v5、NSDI 2011 DRF 原始论文的来源和适用边界记录于 research/distributed-training-systems-2026.md。公开 API 不支持目标集群吞吐或峰值承诺；动态文档应随部署版本复核。
- **code_status**：11 个 Python 3 标准库示例均标记“静态审阅，未运行（2026-10-30）”，紧邻代码块写明保存/运行方法和手算预期输出。10.11 manifest 示例将难手算的 SHA 前缀输出改为三种提交门禁布尔结果，并静态审阅。未运行书中代码、测试、模型、benchmark 或 PDF。
- **formula_status**：复核 global batch、DP/TP/PP world-size 乘积、ring 通信量/单位换算、TP shape、pipeline bubble 假设、ZeRO/FSDP 状态和瞬时峰值、累积梯度分母、tokens/s、MFU 的 F_model/T_wall/F_peak、dominant share 和 SLO burn rate。全章 0 个 operatorname、0 个未转义美元分隔符。
- **visual_status**：新增 22 行主讲导航表；保留任务路径和 pipeline schedule 的 text 图示。核心关系已有公式、逐步账本与现有状态轨迹，本轮未新增 Mermaid 或图片。
- **link_status**：92 个 Markdown 链接，53 个本地目标全部存在，39 个外链；新增官方关键来源已定向阅读，原有外链全量终审留待第四册收口。118 个围栏标记（59 对）闭合，其中 11 个 Python 块状态标记紧邻代码块。没有内部 progress/ 路径；git diff --check 通过。
- **sync_status**：简介第 10 章摘要已有准确概述，目录结构不变。GLOSSARY_EN_ZH.md 增加 Collective Order、Distributed Checkpoint、Kubernetes Device Plugin、Kueue Admission、DRF 与 OpenTelemetry event 术语；EXERCISES.md 增加 GPU device plugin—Kueue—scheduler 职责边界题；PAPERS.md 将 PyTorch 分布式框架资料更新到 2.15 并补 DRF 原始论文入口。PDF 未更新。
- **open_risks**：11 个示例均未执行；39 个外链尚未完成第四册册级全量终审。第四册第 11—19 章、整册连续复读和最终门禁仍待处理，因此不标记第四册完成。

## 第 11 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/11-长上下文rag与agent.md`，保留 11.1—11.25 速查结构。按第四册百科契约，章节提供定义、关键数量关系、风险和跨册入口，不重复书 02、07、17、20、22 的完整主讲。章首新增覆盖所有条目的主讲导航表。
- **chapter_scan**：检查长窗口预算、位置利用与压缩、检索/重排/RRF、证据状态和引用、Agent 循环与规划、工具 schema/副作用/幂等、MCP/A2A 版本对象、记忆、代码与浏览器 Agent、多 Agent 协作、trace/replay、评测、安全以及成本/延迟。定向核对 MCP 2026-07-28、A2A v1.0.1、OWASP SSRF/路径穿越和 Anthropic 工程文章；范围为章节主张所需证据，不代表协议实现、安全或 Agent 评测的穷尽审计。
- **candidate_disposition**：不新增或拆分百科章节。将上下文预算变量明定为最终 token 序列中的互斥账本，避免历史与证据重复计算；将 `H_kv` 定义为每层每 token 每个 key 或 value 向量的元素数；区分注意力算术复杂度和物化分数矩阵内存；将 lost-in-the-middle 效用写为证据位置 `u(p_e)`。长上下文位置示例改称测试计划并声明位置槽是抽象槽位，非 tokenizer token。RAG 成功率改为条件链式分解，不引入独立假设。MCP 按当时官方规范修正无状态请求、逐请求能力协商及可选扩展，明确 Host 授权职责；A2A 固定 v1.0.1 规范、wire version 1.0 和 v1.0 迁移边界。工具安全补充 SSRF 目标约束与路径穿越证据，保留文件竞态等实现风险。
- **length_status**：按解释性汉字统计；各入口均为支持型速查，完整主讲由导航表承接，不按独立教程下限验收：

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 11.1 长上下文与位置利用 | 1,714 | 支持型速查 |
| 11.2 上下文工程 | 743 | 支持型速查 |
| 11.3 RAG 总体架构 | 574 | 支持型速查 |
| 11.4 检索与证据质量 | 820 | 支持型速查 |
| 11.5 证据状态与 Agentic RAG | 920 | 支持型速查 |
| 11.6 Grounding 与引用 | 520 | 支持型速查 |
| 11.7 Agent 抽象 | 615 | 支持型速查 |
| 11.8 规划与 workflow 边界 | 418 | 支持型速查 |
| 11.9 工具契约 | 559 | 支持型速查 |
| 11.10 工具执行 | 688 | 支持型速查 |
| 11.11 MCP 协议边界 | 696 | 支持型速查 |
| 11.12 MCP 安全生命周期 | 367 | 支持型速查 |
| 11.13 A2A | 649 | 支持型速查 |
| 11.14 Memory | 447 | 支持型速查 |
| 11.15 Code Agent | 600 | 支持型速查 |
| 11.16 Browser 与 Computer Use | 375 | 支持型速查 |
| 11.17 Multi-Agent | 432 | 支持型速查 |
| 11.18 系统组合边界 | 437 | 支持型速查 |
| 11.19 Trace 与 Replay | 393 | 支持型速查 |
| 11.20 评估 | 619 | 支持型速查 |
| 11.21 安全与提示注入 | 582 | 支持型速查 |
| 11.22 成本与可靠性 | 567 | 支持型速查 |
| 11.23 端到端设计 | 468 | 支持型速查 |
| 11.24 资料与证据 | 414 | 支持型速查 |
| 11.25 统一理解 | 260 | 支持型速查 |

- **example_status**：静态手算/逻辑复核位置槽 `[(0,0),(0.25,25),(0.5,50),(0.75,75),(1,100)]`、RRF 两个排名列表与稳定融合序列 `B,A,D,C`、重复候选拒绝、严格 schema/畸形 JSON/不一致 schema 拒绝、指数退避上限 `[1,2,4,4,4]`、初始调用后的重试 budget，以及根目录路径穿越拒绝。位置槽仅展示索引映射，不是 token 位置实验。预期值是人工推导，不是程序输出。
- **evidence**：MCP 规范写明 JSON-RPC 2.0、自包含无状态请求、逐请求能力协商，以及 Server/Client 核心能力和可选扩展；旧版 lifecycle 与 Roots 语义不泛化到所有当前实现。A2A release 记录列出 1.0.1，规范协商使用 Major.Minor wire version 1.0；v1.0 移除了 Part/流事件中的内嵌 `kind` 并调整枚举，AgentInterface 声明 transport binding 与 `protocolVersion`，HTTP 使用 `A2A-Version`。OWASP SSRF 指南支撑 allowlist、限制重定向、网络层出口控制及解析到私网地址的风险；路径穿越资料展示 `../` 与绝对路径；Anthropic 页面对 2024 tooling landscape 的变化提醒仅作时效边界。来源日期、阅读范围和外推限制见 `research/agent-tool-calling-security-2026.md`。
- **code_status**：5 个 Python 3.8+ 标准库块均带紧邻状态行、保存/运行命令与手算预期输出，标为“静态审阅，未运行（2026-10-30）”；代码没有运行。没有执行章节代码、测试、模型或 benchmark。
- **formula_status**：复核输入预算各互斥 token bucket、`H_kv=n_kv_heads×d_head`、单头 `n×n` 分数矩阵及 `O(n²d)` 计算 / `O(n²)` 中间内存、位置效用 `u(p_e)`、RRF 排名分母、RAG 条件链式概率。`P(success)=P(R)P(C|R)P(G|R,C)P(A|R,C,G)` 不需要独立假设；各分母与变量条件就地解释。0 个 operatorname、0 个未转义美元分隔符。
- **visual_status**：新增 18 行主讲地图表，跨 25 个小节汇总导航；保留现有任务链路 text 图。没有新增 Mermaid 或图片。
- **link_status**：80 个 Markdown 链接，53 个本地文件目标全部存在、27 个外链；32 个 fenced block（64 个标记）配对闭合，其中 5 个 Python；5 个状态行均紧邻代码块。无本地锚点、无 `progress/` 路径；`git diff --check` 通过。新增来源做了定向核验，第四册其余外链全量终审留待册级收口。
- **sync_status**：`PAPERS.md` 增 A2A 官方规范/迁移/release 入口并修复连续编号；`GLOSSARY_EN_ZH.md` 增 Reciprocal Rank Fusion、Evidence Span、MCP、A2A；MCP/A2A 与安全来源事实写入对应 research 文件。`EXERCISES.md` 已覆盖长上下文、RAG、Agent 协议和安全边界，不重复加题。简介第 11 章摘要准确，目录结构无需变化。
- **open_risks**：5 个代码示例未执行；27 个外链未完成册级全量核验。第四册第 12—19 章、整册连续复读和最终门禁待处理，不标记第四册完成。没有更新 PDF。

- **follow_up**：第 12 章连续复读时回看 11.1 的加权上下文分数，补充每个重要性权重必须有限且非负；此前仅要求权重和为正，未排除负权重导致“加权平均”超出预期范围。该修正不改变章节边界、代码或配套内容。

## 第 12 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/12-reasoning与评估.md`，保留 12.1—12.17 条目结构和第四册速查定位。新增 13 行主讲导航，将 reasoning、CoT、verifier、search、TTC 和 math/code reasoning 接入第十六册及第二册，将统计评估、污染、人评、judge 和发布评估接入第七册，并按主题连接训练、Agent 安全和 Harness 评估章节。
- **chapter_scan**：覆盖答案/状态/候选、CoT 与可见解释、Self-Consistency、Verifier 校准和 hard negative、PRM/ORM、搜索与 UCT、测试时预算、数学合成数据、代码 pass@k、paired lift/bootstrap、benchmark 污染、LLM judge、人评、factuality/grounding、reasoning safety、发布证据向量和 harness-aware evaluation。定向核对 CoT、Self-Consistency、Training Verifiers、Let's Verify、ToT、HumanEval、校准、HELM 和 MT-Bench 论文，并连接已经有证据记录的 2025–2026 TTC、FACT-E 与评估激励研究；论文主张不外推到所有任务和运行环境。
- **candidate_disposition**：不新增或拆分章节。定义 pairwise ties 为半分并固定候选优劣方向；为 ECE/Brier、步骤指标、paired lift/bootstrap、公开/隐藏测试、鲁棒性和关键切片补充分母、范围或 N/A 条件。把 pass@k 的 k 限定为正整数，并说明公式假设同一策略的独立样本及组合数边界。修正 bootstrap helper 将规范化后的输入 tuple 返回，移除“置信区间必须包含点估计”的不成立断言；只输出确定的 lift 与端点次序，不对随机抽样端点作未核验承诺。区分 token、FLOPs、墙钟时间和费用单位，避免预算账本混加。
- **length_status**：按解释性汉字统计；12.1—12.15 与 12.17 均作为支持型百科速查，主讲位置见导航表：

| 入口 | 汉字 | 类型 |
| --- | ---: | --- |
| 12.1 推理对象 | 523 | 支持型速查 |
| 12.2 CoT 与可见解释 | 580 | 支持型速查 |
| 12.3 Self-Consistency | 538 | 支持型速查 |
| 12.4 Verifier | 742 | 支持型速查 |
| 12.5 过程监督 | 599 | 支持型速查 |
| 12.6 Search Reasoning | 547 | 支持型速查 |
| 12.7 Test-Time Compute | 463 | 支持型速查 |
| 12.8 数学推理训练 | 439 | 支持型速查 |
| 12.9 Code Reasoning | 553 | 支持型速查 |
| 12.10 评估样本与统计比较 | 460 | 支持型速查 |
| 12.11 Benchmark 与污染 | 472 | 支持型速查 |
| 12.12 Human Evaluation 与 Judge | 473 | 支持型速查 |
| 12.13 Factuality、Robustness 与 Calibration | 341 | 支持型速查 |
| 12.14 Reasoning Safety | 404 | 支持型速查 |
| 12.15 Evaluation Pipeline | 317 | 支持型速查 |
| 12.17 统一理解 | 218 | 支持型速查 |

12.16 是资料与证据索引，不作为教学单元计篇幅；统计不含标题、表格、围栏内容、公式、状态行和参考条目。
- **example_status**：多数投票候选 `42,42,41` 的胜出比例为 2/3；paired bootstrap 输入 5 对结果的配对 lift 为 0.2。第二个示例使用固定随机种子计算 bootstrap 区间，但输出只验证 lift 和区间形状，不断言特定区间数值。所有输出均由静态逻辑复核，不是运行结果。
- **evidence**：CoT 原论文实验是三种模型和特定算术/常识/符号任务；Self-Consistency 有独立研究记录。Training Verifiers、Let's Verify 和 ToT 证据分别限于 GSM8K/MATH 及论文规划任务。HumanEval §2.1/Appendix A 支持 pass@k 的 n/c/k 无偏估计；Guo ECE、HELM 多指标和 MT-Bench 评审偏差都按原论文范围描述。Snell TTC 分配、Hariri arXiv v2 taxonomy、FACT-E 和准确率/拒答激励边界见 `research/test-time-compute-and-evaluation-2026.md`；新核对的基础论文和证据等级也追加到该记录。
- **code_status**：2 个 Python 3.8+ 标准库示例均有紧邻状态、保存/运行方式及确定输出，标注“静态审阅，未运行（2026-10-30）”。代码未执行；未运行测试、模型、benchmark 或 PDF。
- **formula_status**：复核 CoT regression、temperature softmax、self-consistency、pairwise tie 计分、ECE/Brier、first-error、step/auto coverage、UCT、TTC 预算向量、cost per correct/marginal gain、pass@k、public-hidden gap、paired lift/bootstrap、robustness drop、judge agreement 与 severity-weighted rate。显式处理空样本/空切片、正预算、tie 和模型/任务条件。0 个 operatorname、0 个未转义美元分隔符。
- **visual_status**：新增 13 行跨册导航表。已有请求链路 text 图与发布证据向量，不新增 Mermaid 或图片。
- **link_status**：58 个 Markdown 链接，34 个本地目标均存在、24 个外链；39 个 fenced block（78 个标记）闭合，2 个 Python 状态行均紧邻代码块。无 `progress/` 路径；`git diff --check` 通过。新增来源已读摘要及相关公式/偏差段，余下外链终审留到册级收口。
- **sync_status**：`PAPERS.md` 将 reasoning taxonomy 固定到 arXiv v2；`GLOSSARY_EN_ZH.md` 明确 verifier pairwise accuracy 的 tie 计分约定。评估练习已有相关覆盖，无需重复增题。2025–2026 TTC、FACT-E 和 accuracy/abstention evidence 已出现在 `12.16`，来源事实记录在 research。简介摘要准确，目录不变。
- **open_risks**：两个 Python 示例未执行；24 个外链未完成第四册全量终审。第 13—19 章、整册连续复读和册级最终门禁待处理，不标记第四册完成。PDF 未更新。

## 第 13 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 chapters/13-安全与治理.md 的 13.1—13.27；保持第四册百科速查定位和条目顺序。章首新增 27 行逐条导航，分别连接第八册安全专题、第七册评估、第五册数据工程、第十七册 Agent、第 20 册 Harness 和第二十三册 AI Infra 主讲。
- **chapter_scan**：覆盖威胁模型与风险排序、对齐和代理目标、可扩展监督、HHH、策略层级、jailbreak/间接注入、taint、拒答校准、个人信息与成员推断、DP-SGD、数据生命周期、水印/C2PA、红队、评估统计、事故响应、机制可解释性、steering、模型编辑与遗忘、工具和 Agent 安全、模型/系统卡、公平与毒性、guardrail、供应链以及发布约束。修整若干评估分母条件的插入方式，使定义、N/A 条件和分母示例接在对应指标正文中。无新增、拆分或迁移章节。
- **candidate_disposition**：所有 13.1—13.26 保留为支持型百科条目，由导航表承接专题主讲；13.27 是资料索引，不作教学单元。更新 OWASP LLM Top 10 到 2026 分类，固定 C2PA 2.5、SLSA 1.2、NIST SP 800-218/218A，并注明 AI RMF 1.0 在 NIST 页面所示的修订状态。
- **length_status**：按解释性汉字统计；去除标题、围栏、表格、状态行后，13.1—13.26 均为支持型速查并链接完整主讲，13.27 是参考索引、不计教学篇幅。

| 条目 | 汉字 | 类型 |
| --- | ---: | --- |
| 13.1 威胁建模 | 946 | 支持型速查 |
| 13.2 对齐与目标错配 | 801 | 支持型速查 |
| 13.3 可扩展监督 | 652 | 支持型速查 |
| 13.4 HHH | 550 | 支持型速查 |
| 13.5 策略与指令层级 | 469 | 支持型速查 |
| 13.6 Jailbreak 与 Prompt Injection | 590 | 支持型速查 |
| 13.7 数据流与 taint | 322 | 支持型速查 |
| 13.8 拒答校准 | 440 | 支持型速查 |
| 13.9 隐私与训练数据提取 | 654 | 支持型速查 |
| 13.10 DP-SGD | 507 | 支持型速查 |
| 13.11 数据治理 | 659 | 支持型速查 |
| 13.12 水印与 C2PA | 467 | 支持型速查 |
| 13.13 红队评估 | 435 | 支持型速查 |
| 13.14 安全评估统计 | 554 | 支持型速查 |
| 13.15 事故响应 | 510 | 支持型速查 |
| 13.16 Mechanistic Interpretability | 420 | 支持型速查 |
| 13.17 Steering | 378 | 支持型速查 |
| 13.18 Model Editing | 500 | 支持型速查 |
| 13.19 Machine Unlearning | 359 | 支持型速查 |
| 13.20 Tool Safety | 414 | 支持型速查 |
| 13.21 Agent Safety | 398 | 支持型速查 |
| 13.22 Model/System Card | 473 | 支持型速查 |
| 13.23 偏见、公平与毒性 | 1,225 | 支持型速查 |
| 13.24 内容审核与 Guardrail | 900 | 支持型速查 |
| 13.25 模型与软件供应链 | 936 | 支持型速查 |
| 13.26 能力与安全取舍 | 316 | 支持型速查 |

- **example_status**：加入或补全可复算例子，包括威胁排序的 60 对 48、代理引用分数的 2 对 0、高风险 FNR 2/20、文本/系统 ASR 5/20 与 1/20、拒答率 10% 与 4%、匹配成员/非成员平均损失差 0.20、PII 脱敏召回率 9/10、DP-SGD 裁剪与一次假设噪声样本的更新 (0.4, 0.6)、最小化率 4/5、水印 z=2、Wilson n=10/k=0 区间约 [0, 0.278]、SWR 62.5%、公平选择率差 0.2、guardrail 阈值路由、以及约束优化中 A 不可行而 B 可行。示例值均为教学输入，不代表生产测量。
- **code_status**：3 个 Python 3.8+ 标准库块分别位于 13.1 风险排序、13.7 taint 传播、13.16 activation patching；均有紧邻“静态审阅，未运行（2026-10-30）”状态和命令。手算预期分别为风险分 60/48/30 并拒绝 0 级输入；标签并集含三个来源且 may_send_external 为 False；patch 分数为 -2/0/2。代码未执行，未运行测试、模型、benchmark 或 PDF。
- **formula_status**：静态核对离散风险分排序边界、期望成本、损失函数与分布相关性、FNR/ASR/拒答率、membership inference 集合平均损失、DP-SGD 裁剪与噪声、字段最小化率、简化水印 z 分数、完整 Wilson 区间、严重度加权风险、群体选择率/TPR、steering、编辑/遗忘目标、工具 Allow 合取、guardrail 阈值和风险约束优化。空分母均按 N/A 处理；0 个 operatorname、0 个未转义美元分隔符。
- **visual_status**：新增导航表；威胁模型使用资产/主体/攻击面/影响清单，注入攻击用分阶段列表，评估结果和系统控制用表格/状态关系表达。考虑到这些顺序关系已由紧凑清单与主讲链接承接，本章不重复添加流程图。
- **evidence**：当日定向核对 NIST AI RMF 1.0 官方状态、NIST AI 600-1、OWASP LLM Top 10 2026 与 Agentic Top 10 2026、C2PA 2.5、NIST SP 800-218/218A 和 SLSA 1.2 官方原始页面/规范。OWASP 2026 PDF 版本历史的发布日期仍是占位文本，因此只采用资源页日期，并记录这一边界。来源事实及版本、阅读范围和证据限制保存在 research/ai-safety-governance-standards-2026.md；OWASP 2025 历史口径与 2026 更新记于 research/hallucination-evaluation-and-safety-2026.md。
- **link_status**：76 个 Markdown 链接，49 个本地引用目标均存在、27 个外链；45 对 fenced block 配对闭合；3 个 Python 状态标记均紧邻代码块。无 progress/ 内部路径；git diff --check 通过。新增的标准/分类官方来源已定向阅读，其余旧论文和动态外链全量终审留待第四册收口。
- **sync_status**：GLOSSARY_EN_ZH.md 将 jailbreak 分类更新为 OWASP LLM01:2026 direct prompt injection 子类。新增标准与版本证据研究记录；原有 EXERCISES.md 已覆盖安全、红队、提示注入、权限和 Agent 安全，不重复增题。PAPERS.md 不纳入行业清单和标准；简介、目录摘要仍准确。
- **open_risks**：3 个 Python 示例未执行；27 个外链未完成第四册全量终审。C2PA/OWASP/NIST/SLSA 动态版本应在册级收口时复查；NIST AI RMF 1.0 与 OWASP PDF 的版本发布日期边界已明示。第四册第 14—19 章、整册连续复读和最终门禁待处理，不标记本册完成；PDF 未更新。

## 第 14 章逐章扫描与静态精修（2026-10-30）

- **scope**：从头至尾审阅原 947 行第 14 章。保留 14.1—14.46 章节入口和“重要论文索引”职责，把章节从论文讲义收束为代表性论文导航。每个索引条目保留研究问题/主要改变、证据边界和原始来源；章首按主题链接第十册论文方法及书系机制、训练、部署、推理、多模态、安全和评估主讲。
- **chapter_scan**：检查架构与位置、规模与训练、后训练与 RLVR、参数适配与分布式系统、RAG 与工具、测试时计算、多模态、安全/编辑/解释性、Agent benchmark 等主题的入口完整性、重复和版本。论文阅读方法原由 14.1 及代码演示承担，与第十册第 2—6 章重复；14.42 横向比较与第四册第 15 章重复；逐篇公式和较长机制讲解分散在相邻书册主讲中。正文定位重叠的段落改为研究结论速查和明确回链，不另复制教程。
- **candidate_disposition**：保留代表性经典论文和主题覆盖，不把“重要”解释为穷尽或排行榜。新增到仓库级 PAPERS.md 的动态补充入口，使 2026 年后续候选论文和官方报告可持续收录而不稀释本章稳定索引。第 14.42 改为指向第 15 章；第 14.46 单独标出官方 API/协议资料为非论文证据。
- **length_status**：本章 14.2—14.41 均为支持型论文条目，不承担完整机制教学；14.1、14.42—14.46 为使用方式、导航、记录模板和资料入口。第十册及各专题书册承接连续教学；因此不把逐篇短条目误计成独立辅助讲解单元。
- **example_status / code_status / formula_status**：索引不适用数值教学和可执行示例。移除原 14.1 的教学成本账本代码以及多处脱离主讲上下文的局部公式，改由相关章节提供完整推演。当前无代码块、公式围栏或行内公式，不需要运行章节代码。
- **evidence**：检查条目所列来源类型与论文主张是否匹配，并按模型/数据/预算限定结果。定向阅读 DeepSeek-R1 arXiv v2（修订 2026-01-04）、DAPO v2（2025-05-20）、Dr.GRPO v2（2025-10-06）元数据与摘要，OpenAI 当前 function calling 官方指南、MCP 2026-07-28 规范，以及 GPT-2 原始 OpenAI 论文；章节补充 GPT-2 直达链接并固定 R1/DAPO/Dr.GRPO 版本。其余外链语法和本地目标静态检查通过，完整外链内容终审留在册级门禁。
- **visual_status**：新增 10 行书系主题导航表，按论文条目组别连接对应主讲；无复杂数据流或算法图示需求，不新增 Mermaid/图片。
- **link_status**：107 个 Markdown 链接，其中 49 个本地目标全部存在、58 个外链；0 个 fenced code block；无 progress/ 内部路径。git diff --check 通过。
- **sync_status**：PAPERS.md 中 DeepSeek-R1、DAPO、Dr.GRPO 的引用固定到已核对版本；research/test-time-compute-and-evaluation-2026.md 新增 DAPO/Dr.GRPO 来源、版本、作者报告和边界。OpenAI function calling 与 MCP 来源证据已在 research/agent-tool-calling-security-2026.md 留存。简介第 14 章摘要仍准确，目录无需调整；题库、练习、术语表无需复制论文索引。
- **open_risks**：本轮重点核验时效性与章节锚点，不等于第四册所有外链全量终审；余下旧链接将在第四册册级门禁统一检查。整册连续复读与最终验收仍待完成。未运行章节代码、测试、benchmark、模型或 PDF。
## 第 15 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文审阅 `chapters/15-面试高频对比表.md` 的 15.1—15.43，保留“高频横向复习与专题导航”定位。第 15.1—15.42 是受限比较条目，第 15.43 是来源与证据入口；没有把本章改成替代主书的课程。保留并复核了架构审视中提到的用户工作区说明：FlashAttention 的 dense 目标不等于跨 kernel bitwise 相同，稀疏 mask 改变相对全连接目标的配对集合。
- **chapter_scan**：检查训练优化、架构、对齐、解码、微调/压缩、注意力与 serving、检索、Agent、评估和安全比较是否回答共同问题、机制、代价、失效边界和验证口径；核对研究结论的版本/实验范围、公式域、可复算例子、代码状态及主讲入口。没有发现需要新增或拆分的独立比较主题；增加了按主题通往书 3、5—10、16、17、21 的 18 行主讲导航。
- **candidate_disposition**：保留现有比较条目。它们只总结局部差异并提供选型维度，不承担可独立学习的完整机制课；专题推导、实战和系统测量由导航表所列章节负责。没有复制专题书的长讲解，也没有新造术语表/练习主题。
- **length_status**：15.1—15.42 均为第四册工具书中的支持型比较速查，逐项字数如下（统计解释性汉字；排除标题、表格、代码/公式围栏、状态行；它们不是独立教学单元，主讲入口见章首导航）。总计 12,155 汉字。15.43 是来源索引，不计讲解单元。

| 条目 | 汉字 | 条目 | 汉字 | 条目 | 汉字 |
| --- | ---: | --- | ---: | --- | ---: |
| 15.1 | 626 | 15.2 | 452 | 15.3 | 311 |
| 15.4 | 280 | 15.5 | 245 | 15.6 | 184 |
| 15.7 | 288 | 15.8 | 241 | 15.9 | 226 |
| 15.10 | 259 | 15.11 | 196 | 15.12 | 320 |
| 15.13 | 344 | 15.14 | 364 | 15.15 | 267 |
| 15.16 | 207 | 15.17 | 328 | 15.18 | 232 |
| 15.19 | 153 | 15.20 | 295 | 15.21 | 239 |
| 15.22 | 187 | 15.23 | 236 | 15.24 | 227 |
| 15.25 | 278 | 15.26 | 236 | 15.27 | 286 |
| 15.28 | 388 | 15.29 | 346 | 15.30 | 281 |
| 15.31 | 551 | 15.32 | 373 | 15.33 | 297 |
| 15.34 | 287 | 15.35 | 253 | 15.36 | 216 |
| 15.37 | 254 | 15.38 | 207 | 15.39 | 329 |
| 15.40 | 239 | 15.41 | 264 | 15.42 | 363 |

- **example_status**：加入同一组 100 个请求上的 Pareto 手算：A 为 94/100、0.03 元/请求、p95 1.4 秒；B 为 92/100、0.02 元、0.8 秒，二者均可能非支配；C 为 91/100、0.025 元、1.2 秒，被 B 支配。静态复核 Adam 首步结果约 0.6838/0.6738；补充负二阶矩拒绝分支。另静态复核 temperature/top-p 分布、greedy 与 beam 序列概率、FSDP 理想状态份额 80/8=10 GB/rank、retriever×reranker 的 0.90×0.70=0.63、pass@5 组合数和危险漏判/安全误拒分母。15.42 客服知识库是假设教学场景，所有构造数字均非生产测量。
- **evidence**：对照阅读 `arXiv:2407.16833v2` 摘要、引言和 §3.1—§3.3：9 个 LongBench/∞Bench 英语 query 数据集、三种指定模型与检索设置下的作者结果，同时保留 GPT-3.5-Turbo 在两个超窗数据集上的例外；阅读 `arXiv:2501.01880v1` 的摘要、筛题/扩数据说明和结果，限定其 QA、摘要式/分块检索与对话任务结论。检查 Anthropic 2024-12-19 页面中的 workflow/agent 操作性区分和当前工具生态过时提示。CoT v2、Judge v4、FActScore v2 的来源范围一并记录于 `research/method-comparison-evidence-2026.md`；AdamW v3 证据沿用既有优化器研究记录。没有把论文比较外推成当下所有产品成本或统一排名。
- **code_status**：唯一 Python 示例位于 15.2，使用 Python 3 标准库 `math`、`numbers`，给出保存/运行方法、手算预期输出和“静态审阅，未运行（2026-10-30）”状态。新增对负二阶矩状态的拒绝及对应分支；代码未运行，预期输出不是运行结果。本轮没有运行程序、测试、模型或 benchmark。
- **formula_status**：静态复核 Adam/AdamW 更新及首步数值、top-k/top-p 累计概率、温度示例、GQA/KV 容量、并行 batch、状态分片、检索条件概率、总成本账本、pass@k、FNR/FPR 等。补足 top-k 的 `1≤k≤V`、温度 `T>0`、top-p `p_cut∈(0,1]`，说明过滤后重归一化及 top-k/top-p 组合顺序需按实现记录。稀疏 mask 的精确目标和 dense 配对集差异仍明确区分。0 个 `operatorname`、0 个美元数学分隔符。
- **visual_status**：新增 18 行跨册导航表；保留 Pareto、训练状态、检索链路和混淆矩阵对照表。全章 35 个 math 围栏和 1 个 Python 围栏；无 Mermaid 或图片需求，比较关系已由导航表/对照表承载。
- **link_status**：126 个 Markdown 链接，其中 41 个本地目标全部存在、85 个外链；36 个围栏块（35 math、1 Python）配对闭合；无 `progress/` 正文路径。新增/固定的 RAG、Anthropic、CoT、Judge 与 FActScore 来源已定向阅读；`git diff --check` 通过。
- **sync_status**：`PAPERS.md` 将 AdamW 固定到 v3、FActScore 固定到 v2、RAG/LC 两篇固定到 `2407.16833v2` 与 `2501.01880v1`、Judge 固定到 v4、CoT 忠实性论文固定到 v2。新研究证据和适用边界写入 `research/method-comparison-evidence-2026.md`。没有新增需要更新的术语定义或练习；简介第 15 章摘要仍准确，目录结构未变。
- **open_risks**：Adam/AdamW 示例未执行；85 个外链中，本轮完成新增重点来源定向阅读，其他来源仍待第四册册级链接与证据终审。正文 15.1—15.42 作为支持型比较条目的判断依据是本册工具书契约及明确主讲导航；不作为 42 个独立完整讲解课验收。第四册第 16—19 章、整册连续复读和最终门禁未完成，不能标记本册完成。PDF 未更新；下一步处理第 16 章《多模态学习》。
## 第 16 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文复读原 1,424 行《多模态学习》，保留 16.0—16.31 查询入口，将章节重组为第四册速查和书系导航。新增 10 行跨册导航，连接第十五册多模态主讲、本册视觉生成/语音/安全条目，以及第五、六、十一、十八册相关章节。现稿 369 行。
- **chapter_scan**：检查模态表示、采样与 token 预算、输入保真和证据召回、CLIP/SigLIP、视觉编码器与连接器、VLM 指令微调、扩散/视频/音频、统一多模态、评估、安全、事故复盘和产品路径。图像/视频生成、语音、统一架构与评估安全原有长篇机制和第十五册第 06—11、13—14 章及本册第 17—18 章重叠；压缩成定义、关键边界、代表指标和直接主讲入口。跨模态证据链与输入/上下文预算保留为本章查询主线。
- **candidate_disposition**：不新增、拆分或迁移章节。16.1—16.14 继续提供表示、对齐、VLM、连接器和指令微调的概念入口；16.15—16.21 分别指向视觉生成、视频与语音主讲；16.22—16.28 保留统一路线、评估、安全、事故和产品账本速查。完整推导沿章首导航进入第十五册、第五/六/十一/十八册专题。
- **length_status**：按解释性中文字符计数，排除标题、表格、围栏代码/公式、状态标记；编号单元均为有主讲链接的支持型百科入口，不承担独立完整教程，故不按 1,000/1,800 字教学单元门槛验收。逐项计数：16.1 141、16.2 218、16.3 160、16.4 112、16.5 206、16.6 240、16.7 113、16.8 147、16.9 98、16.10 86、16.11 87、16.12 102、16.13 109、16.14 181、16.15 143、16.16 69、16.17 155、16.18 170、16.19 167、16.20 162、16.21 149、16.22 157、16.23 126、16.24 216、16.25 192、16.26 73、16.27 193、16.28 127、16.31 293。16.0 引言/导航为 200；16.29 是来源索引（209，含来源注释），16.30 是练习题（208，含题目），均非讲解单元。
- **example_status**：复算 16 kHz × 60 秒 = 960,000 个采样点；336×336 图像、patch=14 得 576 个视觉 token，双倍高宽得 2,304 个，序列 token pair 数约增至 16 倍；四字段票据例的输入保真为 3/4、上下文证据召回 2/4、claim 支持率 2/3；CLIP 构造 logits 8/2 得正例概率约 0.9975；WER 构造例为 (1+1+2)/20=20%。示例值均为教学构造输入。代码例预期仅打印 multimodal audit toy: ok，由静态逐句推演得出。
- **evidence**：定向阅读 [Image-based Prompt Injection, arXiv:2603.03637](https://arxiv.org/abs/2603.03637) 的摘要、引言、IV-A 和 IV-C。论文报告在 COCO 单图、无附加文字上下文、GPT-4-turbo 和指定嵌入配置下的最高 64% 攻击成功率；正文限定于该实验，不外推为总体风险率。来源、方法范围和证据边界记录于 research/hallucination-evaluation-and-safety-2026.md。C2PA 2.5 技术规范已在 research/ai-safety-governance-standards-2026.md 记录签名/来源验证与内容真实性的边界。
- **code_status**：唯一 Python 示例的可执行语句未改动；补充“静态审阅，未运行（2026-10-30）”、Python 3.9+ 标准库、保存/运行命令和静态预期输出。手算 token 数 660、claim support 1.0、应拒样本漏判率 0.5，空 claim 和无应拒样本返回 None。未运行章节代码、测试、模型、benchmark 或 PDF。
- **formula_status**：静态核对链式条件概率、patch 数/视频长度、token budget 与平方注意力边界、输入保真/证据召回/claim 支持、CLIP 温度 softmax、SigLIP pair logistic、音频 sample/codec 数、WER/CER 和成功成本定义域。7 个 math 围栏和 35 对美元行内公式分隔符配对；0 个 operatorname。
- **visual_status**：新增 10 行跨册导航；保留 6 个关系表（导航、连接器、统一路线、评估切片、账本、产品工作流）以及数值推演。没有增加 Mermaid；主链路由条件概率、数值例和端到端账本表表达。
- **link_status**：69 个 Markdown 链接，53 个本地目标全部存在、16 个外链；8 个 fenced block（7 math、1 Python）配对闭合；无 progress/ 正文路径。C2PA 及图像注入来源完成定向阅读；其余外链留待第四册册级终审。git diff --check 通过。
- **sync_status**：PAPERS.md 将多模态标准项的 C2PA 链接更新到 2.5 技术规范；图像注入论文已有收录。现有 GLOSSARY_EN_ZH.md 与 EXERCISES.md 已覆盖 CLIP、token budget、证据支持、多模态注入、语音和多模态评估，本轮不重复加条目或练习。简介第 16 章摘要与目录仍准确，无需同步。
- **open_risks**：章节代码未经执行；16 个外链中本轮仅定向复核新安全来源与主要标准，完整外链终审仍待第四册收口。第 17—19 章、整册连续复读和最终门禁尚未完成；PDF 未更新。下一步处理第四册第 17 章《图像生成与视频生成》。

## 第 17 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文复读原 823 行 `chapters/17-图像生成与视频生成.md`，保留 17.0—17.23 编号和主题入口，改为百科速查、数值账本及专题路由。现稿 404 行，新增跨册主讲导航，分别指向第十五册扩散基础、Stable Diffusion/DALL·E、视频生成/World Model、评估与安全章节。
- **chapter_scan**：检查生成路线、条件、采样、表示压缩、视频时序与评估条目是否形成可查路径，哪些段落重复主讲、哪些当前信息需要版本/生命周期限定，以及交互式视频评估是否有新的可迁移入口。扫描覆盖 2025-10-30 至 2026-10-30；资料类型包括原始论文/预印本、OpenAI 官方研究说明与帮助中心、C2PA 2.5 规范及现有研究记录。复用扩散和采样证据 `research/diffusion-generation-guidance-and-sampling-2026.md`、视频/world model 证据 `research/video-generation-world-modeling-and-sora-lifecycle-2026.md`、Flow Matching 证据 `research/flow-matching-and-video-generation-2026.md`；本轮重核官方 Sora 停止服务日期，并定向阅读 WBench 与 World-in-Context 原始论文页面。
- **candidate_disposition**：扩散推导、scheduler、DiT/VAE/CFG、DALL·E 和 ControlNet 的完整机制回到第十五册 06—07；视频生成、运动/状态和 world model 回到第十五册 08；FID/FVD、生成安全和 provenance 回到第十五册 11。本章保留对比表、关键算式、可复算例子及失败诊断。Flow Matching 定位为概率路径上的速度场训练框架，不当作与 U-Net/DiT 并列的网络结构；Pyramidal Flow Matching 作为视频路线短条目链接到原论文，其专题展开待第十五册第 08 章周期审视。WBench v1 提供多轮视频世界模型的五维/22 子指标入口；World-in-Context v1 为新近预印本，仅作视角、动作与摄像机联合控制的研究入口，论文结果不外推到真实环境。未发现需改变第四册 19 章骨架的主题。
- **lifecycle_update**：Sora 段落从当前产品说明改为历史研究与服务生命周期记录。OpenAI 官方帮助中心截至 2026-10-30 明确网页/应用于 2026-04-26 停止、API 于 2026-09-24 停止；来源状态更新在 `research/video-generation-world-modeling-and-sora-lifecycle-2026.md`。不把 2024 报告或 2025 system card 当作当前可调用能力。
- **length_status**：依第四册工具书契约，17.0—17.20 是有完整主讲去向的支持型查询单元，不承担独立教程；合并解释性中文字符共 **3,509**，高于整篇核心单元门槛。逐节计数排除标题、表格、公式/代码围栏、行内公式、链接目标和状态标记；17.21 为来源索引，17.22 为练习路由，17.23 为总结，均不另作讲解单元：

| 单元 | 汉字 | 单元 | 汉字 | 单元 | 汉字 |
| --- | ---: | --- | ---: | --- | ---: |
| 17.0 | 245 | 17.1 | 228 | 17.2 | 167 |
| 17.3 | 159 | 17.4 | 126 | 17.5 | 61 |
| 17.6 | 168 | 17.7 | 153 | 17.8 | 98 |
| 17.9 | 82 | 17.10 | 135 | 17.11 | 110 |
| 17.12 | 125 | 17.13 | 142 | 17.14 | 161 |
| 17.15 | 177 | 17.16 | 346 | 17.17 | 211 |
| 17.18 | 212 | 17.19 | 195 | 17.20 | 208 |
| 17.21 | 28 | 17.22 | 55 | 17.23 | 214 |

- **example_status**：手算前向扩散 `ᾱ=.36, x₀=2, ε=-1 → xₜ=.4`，再以 `v=-2.2` 还原 `x₀=2, ε=-1`；DiT patch 从 16 到 64 token 时 dense pair 数从 256 到 4,096（16 倍）；512×512 RGB 的 786,432 个标量映射到 64×64×4 latent 的 16,384 个标量，比例 1/48；CFG 构造向量例 `(1,0),(2,1),g=3 → (4,3)`；16×32×32 个视频位置为 16,384 token、dense pair 为 268,435,456；一维高斯示例 FID 为 2；采纳率调整后的教学延迟为 2,000 ms 与 2,400 ms/成功输出。上述数字均为构造输入，不代表模型或硬件实测。
- **evidence**：保留 DDPM/CFG、少步采样、DiT/LDM、FID/FVD 原始来源和适用边界；交互式评估沿 arXiv:2605.25874v1 限定为 289 个案例、1,058 轮、五个维度及 22 个自动子指标。arXiv:2610.33169v1 的控制方式/结果只按该预印本的 WBench 配置描述。C2PA 2.5 的来源和签名验证不能证明内容陈述真实或已授权，复用 `research/ai-safety-governance-standards-2026.md`。Sora 生命周期定向核验来源为 OpenAI Help Center，发布日期与阅读位置见研究记录。
- **code_status**：保留唯一 Python 标准库示例的全部可执行语句；补充 Python 3.9+、保存/运行方法、手算预期和“静态审阅，未运行（2026-10-30）”状态。基础配置 latent_cells=4,096、延迟/采纳率=2,000 ms；快速配置为 2,400 ms；零采纳率触发拒绝分支。未运行代码、测试、模型、benchmark 或 PDF。
- **formula_status**：13 个 math 围栏、1 个 Python 围栏，全部配对；检查扩散闭式加噪、噪声/x0/v 互换、反向均值适用范围、CFG 两种系数约定、latent 元素比例、patch/token pair、身份/闪烁代理、FID 与成功成本定义域。76 对行内美元公式；无 `operatorname`、无 progress 路径。
- **visual_status**：新增五行专题导航和多张差异/证据表；保留扩散、VAE、attention、视频 token、编辑指标和单位成功成本的数值解释。没有 Mermaid；流程关系由正文、公式和路由表呈现。
- **link_status**：全文 55 个 Markdown 链接，13 个本地目标全部存在、42 个外链；14 个 fenced block（13 math、1 Python，28 个围栏标记）配对闭合。第十五册主讲文件、练习 25—27 和安全入口均存在；仓库内没有旧链接锚点指向本章已删除的小节。`git diff --check` 通过。
- **sync_status**：`PAPERS.md` 将 WBench 与 World-in-Context 链接固定到 v1；题库和术语表已含 WBench，练习 25—27 已覆盖扩散、文生图和视频/world model，因此不重复增项。简介第 17 章摘要与目录继续准确；专题入口在章首更新。Sora 官方停止服务日期及来源日期更新在对应 research 记录。
- **open_risks**：代码仅静态审阅；scheduler 使用动态官方文档链接，需在第四册外链终审按目标版本核查。WBench/World-in-Context 为论文协议内的评估或预印本结果，不代表现实世界控制能力。第十五册第 08 章后续审阅时应评估是否把 Pyramidal Flow Matching、WBench 作为深讲/评估入口；第四册第 18—19 章、整册连续复读和最终门禁仍待完成。PDF 未更新。

## 第 18 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文复读原 673 行《语音大模型》，保留 18.0—18.23 全部编号，重组为 292 行百科速查与主讲路由。18.0 增加专题导航，连接第十五册第 09、11 章，本册第 09、11、13 章，第十八册第 08 章及第二十一册第 93 章。完整波形/ASR/codec/TTS 教学回到第十五册第 09 章；评估与安全专题回到第十五册第 11 章。
- **chapter_scan**：2026-10-30 定向检查波形与特征、ASR 指标、Whisper 长音频、语义/声学 token、codec 码率、TTS 与韵律、克隆授权、流式识别、VAD/endpoint、打断取消、端到端语音模型、Voice Agent、评估和成本。复用研究记录 speech-recognition-generation-and-realtime-2026.md；重新打开并核对 Qwen3-ASR v2、mu-bench v1、Qwen3-TTS v1、LP-TTS-Bench v1 原论文页面。研究记录标出页面版本复核日期 2026-10-30。
- **candidate_disposition**：不改第四册 19 章骨架，不增删讲次编号。声学特征、CTC/RNN-T、神经 codec、TTS 和 speech-to-speech 保留定义、关键公式或路线表，完整机制由第十五册 09 主讲；WER/MOS、业务字段和安全评估由第十五册 11 主讲；Agent 状态、安全和多模态产品由第四册 11/13 与第十八册 08 主讲。Qwen3-ASR 动态窗口、mu-bench 的 UER 与 normalizer、Qwen3-TTS 双 token 路线/作者首包数和 LP-TTS-Bench 的韵律维度作为当前版本入口，均限定论文和协议范围。Qwen3.5-Omni ARIA 保留到第二十一册 93 的专题链接。未发现需改变书册边界的候选。
- **length_status**：18.0—18.20 都是有专题主讲链接的支持型百科条目，不承担独立完整教程；整合解释性汉字 2,834。逐条计数（排除标题、表格、公式/代码围栏、行内数学及链接目标）：

| 单元 | 汉字 | 单元 | 汉字 | 单元 | 汉字 |
| --- | ---: | --- | ---: | --- | ---: |
| 18.0 | 118 | 18.1 | 104 | 18.2 | 121 |
| 18.3 | 108 | 18.4 | 175 | 18.5 | 166 |
| 18.6 | 114 | 18.7 | 46 | 18.8 | 111 |
| 18.9 | 139 | 18.10 | 158 | 18.11 | 102 |
| 18.12 | 95 | 18.13 | 145 | 18.14 | 110 |
| 18.15 | 182 | 18.16 | 124 | 18.17 | 147 |
| 18.18 | 148 | 18.19 | 151 | 18.20 | 270 |
| 18.21 | 28 | 18.22 | 51 | 18.23 | 143 |

18.21 是证据与版本索引，18.22 是练习导航，18.23 是总结，不计作独立讲解单元。计数是静态口径，用于长度记录；条目满足百科定位的依据是可查定义、最小直觉/指标例、明确边界和到达完整主讲的路径，不以章节总字数代替教学质量。
- **example_status**：手算 16 kHz×60 秒=960,000 个采样点；WER 示例参考 4 词、替换 1、删除 1，得到 2/4=0.5；20 Hz×30 秒×8 码本得到 4,800 code ID；串行延迟代理 120+180+240+90=630 ms；成本构造例 0.020/0.8=0.025 美元/成功任务。代码预期字段准确率 1/2=0.5、两请求成功率 1/2=0.5、无字段返回 None。
- **evidence**：Qwen3-ASR 按 arXiv:2601.21337v2 的 AuT encoder、动态 1—8 秒 attention window 与作者报告的流式/离线支持描述；不从 AED 类别推导所有系统可流式。mu-bench arXiv:2609.32082v1 的 4,270 utterance/250 通电话/五种 locale、κ=0.78 与 normalized exact-match WER=0.53 绑定论文数据、1,847 条人工评审子集、judge、rubric 和 normalizer。Qwen3-TTS arXiv:2601.15621v1 两种 codec 路线及 97 ms 首包绑定报告配置；LP-TTS-Bench arXiv:2610.19774v1 的约 10 分钟观察不外推成通用 prompt 建议。Whisper 使用 arXiv:2212.04356v1 与 Transformers v5.17.0 固定文档；C2PA 限定为来源/签名凭证，不证明媒体内容真实或声纹授权。
- **code_status**：唯一 Python 标准库代码块与原稿可执行语句逐字比较一致；标记“静态审阅，未运行（2026-10-30）”，声明 Python 3.9+、保存/运行命令与手算输出。未运行该示例、测试、模型、benchmark 或 PDF。
- **formula_status**：6 个 math 围栏覆盖采样点、log-Mel、WER/CER、关键字段、codec token 和单位成功成本；1 个 text 流程围栏、1 个 Python 围栏；围栏 16 个标记/8 对闭合，行内美元数学 22 对；0 个 operatorname；无 progress/ 正文路径。核查 WER/CER 与关键字段的分母、codec q/D/K 定义域、成本分子与成功 cohort 的对应，以及串行阶段相加只作教学代理。
- **visual_status**：新增主讲路由表；表示路线、端到端架构、评估口径和故障阶段使用对照表；barge-in 用最小状态文本流程与 trace 边界说明。无需新增 Mermaid。
- **link_status**：43 个 Markdown 链接，27 个本地目标全部存在、16 个外链；EXERCISES 的阶段锚点存在；目标链接无断链。重点新增/固定论文页面已定向阅读；其他章节外链仍待第四册最终外链门禁。git diff --check 通过。
- **sync_status**：PAPERS.md 将 Qwen3-ASR、mu-bench、Qwen3-TTS、LP-TTS-Bench 固定到 v2/v1 版本链接；研究证据仍在 research/speech-recognition-generation-and-realtime-2026.md，未写入进度事实。GLOSSARY_EN_ZH.md 已有 UER、LP-TTS-Bench、TTFC、RTF、ARIA、WER/CER 术语；EXERCISES.md 已有 28A—28C 覆盖 CTC/RNN-T、业务 WER/UER、韵律评测，不重复添加。简介第 18 章摘要和目录摘要准确，无需改动。
- **open_risks**：Python 示例仅静态审阅；16 个外链未完成第四册整册最终核查；模型测量端点和动态官方实现仍须按部署版本重核。第四册第 19 章、全册连续复读和最终门禁待做。PDF 未更新。
- **follow_up**：连续复读时补充 `C_success` 定义域：`P_task_success>0`；零成功时单位成功成本无定义，报告 N/A。代码未变化。

## 第 19 章逐章扫描与静态精修（2026-10-30）

- **scope**：全文复读原 366 行 `chapters/19-frontier-architecture-updates.md`，整理为 98 行机制索引、评测诊断、来源分级快照和跨册入口。核心作用是帮助读者识别问题落在序列状态、深度聚合、残差流、压缩/索引或路由容量哪一轴；机制和模型专题回链第二十一册及相关主书。
- **chapter_scan**：核对 AttnRes/DCA 深度聚合、KDA/Kimi Linear 递归状态、DeepSeek V4 CSA/HCA 与 Qwen3.8 QSA、mHC/Gated Residual、多值专家路由与 N-gram 查表；复查当前官方披露的 Gemini 4 Argon、Mistral Large 4、GPT-6.1 Terra 和 Claude Haiku 5.5。重点区分论文机制、模型卡、接口字段与系统卡摘要，不用模型名、排行榜或接口能力补写未披露的架构。
- **candidate_disposition**：没有发现需新增、拆分或迁移第四册章节的主线。保留机制轴定义、一个压缩/候选漏选手算、KV cache 账本与故障诊断表；各机制的推导、模型规格和系统实现转到第二十一册第 75—95 章。GPT-6 Astra、GLM-5.3 长任务、Kimi K3 harness 继续可从本章入口到第六册 18、第十六册 20、第十七册 15。MSAR 只据 arXiv 摘要描述方向，不记录效果数字或通用比较结论。
- **length_status**：核心连续解释单元在模型快照前含 1,930 个解释性汉字，排除标题、表格、链接目标、引用和格式标记后计数，达到核心单元 1,800 下限。轴对照、诊断、模型快照与章节导航表均为带主讲入口的支持型索引，不作为独立教程计数。
- **example_status**：教学构造输入 16 个 token，每 4 个压一块，得到 4 个候选；top-2 漏掉排名第三的证据块，预算提到 3 可修复候选漏选；若压缩已丢掉定义限定词，dense read 也不能重建原 token。另手算完整 KV 下界：`L=2,T=16,H_kv=2,d_h=4,b=2 bytes`，每 token/层 K+V 为 32 bytes，16 token 为 512 bytes/层，两层为 1,024 bytes；说明不含对齐、元数据、量化 scale、workspace 和 allocator 开销。所有数值均为教学输入。
- **evidence**：阅读 Attention Residuals 方法说明；ICML 2025 DeepCrossAttention 官方论文页摘要；Kimi Linear v2 方法/结果切片；DeepSeek V4 CSA/HCA 与 serving cache 段；mHC v2 双随机映射、Sinkhorn 与特定开销报告；MSAR 摘要。模型披露使用 Gemini 4 Argon 官方卡、Mistral 模型文档/NVIDIA 口径页、OpenAI GPT-6.1 Terra 发布与 API 页、Anthropic Haiku 5.5 Transparency Hub 摘要和产品页。版本、支持事实与证据边界分别写入 `research/transformer-architecture-research-2026.md` 和 `research/frontier-model-architecture-scan-2026.md`。Mistral 的 49B/52B 按每 token 专家参数与计入 embedding/输出层的口径解释；Haiku 5.5 系统卡 PDF 因文件过大未读取，仅使用公开摘要。
- **code_status**：本章无可执行代码；未运行代码、测试、模型、benchmark 或 PDF。
- **formula_status**：唯一 KV 近似式 `M=2LTH_kv d_h b` 已定义变量、字节单位和忽略项，并手算到 1,024 bytes。正文有 13 对行内美元分隔符；无 math 围栏、`operatorname` 或 HTML 数学标记。
- **visual_status**：机制轴、失败诊断、披露层级和专题去向由四张有说明的表呈现。没有独立时序或系统流程需要图；未新增 Mermaid。
- **link_status**：47 个 Markdown 链接，其中 34 个本地目标全部存在、13 个外链；无代码围栏、无 `progress/` 路径。链接中的模型来源页和论文入口均贴近相关主张；`git diff --check` 在最终记录前通过。
- **sync_status**：更新第四册简介第 19 章摘要和目录入口；`PAPERS.md` 加入 MSAR 摘要级预印本说明、Gemini 模型卡及 Mistral/GPT-6.1/Haiku 官方披露入口；research 只留知识和证据。现有 `GLOSSARY_EN_ZH.md` 已有 KDA、CSA/HCA、AttnRes、mHC、MoVA、QSA、N-gram 等定义；`EXERCISES.md` 已覆盖压缩/索引召回、KV cache、KDA、CSA/HCA 与 MoVA，不重复新增同义条目。
- **open_risks**：MSAR 完整论文与实验表未审；Claude Haiku 5.5 完整 System Card PDF 未读，正文仅使用官方摘要支持的结论；外部链接全量最终门禁尚待第四册连续复读。第四册不能据单章完工标为整册完成。PDF 未更新。

## 第四册整书连续复读（第 01—02 章，2026-10-30）

- **scope**：依简介、目录和主阅读路径，完整连续复读第 01 章《数学与机器学习》与第 02 章《深度学习组件》，并对照两章既有精修记录核对索引定位、主讲分流及章间衔接。
- **continuity_status**：第 01 章是数学、概率、优化和统计评估的速查入口；第 02 章沿 decoder-only Transformer 从张量与组件走到训练、分布式和诊断，链接回第 01 章交叉熵并分流到书 05、14、21 主讲。连续阅读未发现主题边界、标签右移、有效 token、mask 语义、DDP token 加权、公式定义域或代码状态间的新矛盾，本轮无需修改正文。
- **open_risks**：579 个外链的全量可达性与证据适配复核仍未完成；连续复读仅完成第 01—02 章。下一步从第 03 章继续，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 03 章，2026-10-30）

- **scope**：完整复读《NLP 与 Tokenization》，按数据流检查 tokenization、词表、BPE/WordPiece/Unigram、字符/字节、多语言、特殊 token、chat template、上下文窗口、detokenization、迁移和训练/评估单位。
- **continuity_status**：本章以 token id 和序列化协议收束，并明确交接到第 04 章 Transformer 架构；公式、手算例、代码静态状态及算法/接口/实测证据边界衔接一致，未发现需要改正文的阻断项。
- **open_risks**：579 个外链的全量可达性与证据适配复核未完成；连续复读仅完成第 01—03 章。下一步从第 04 章继续，整册门禁保持未完成。

## 第四册整书连续复读（第 04—10 章，2026-10-30）

- **scope**：沿简介与目录顺序完整连续复读第 04—10 章，检查 Transformer 架构、LLM 基础、预训练数据、后训练对齐、解码、推理部署和分布式训练系统之间的输入/状态/训练/服务交接；章节已有精修记录与当前正文一并对照。
- **continuity_status**：第 04 章明确从 token 序列进入模型计算，第 05 章串起训练与产品运行，第 06—07 章以数据契约和目标函数承接训练阶段，第 08—09 章从解码进入服务生命周期，第 10 章再扩展到分布式计算、集群调度及训练平台。复杂度、标签 mask、算法比较、概率和成本分母、请求状态、KV 生命周期及系统边界在连续主线中未出现新的冲突。本轮连续阅读发现并修正第 04 章单头/多头复杂度说明，以及第 09 章非法清理调用会提前清零 KV 计数的状态 bug；第 10 章 AllReduce、流水线气泡、分片显存、MFU、SLO burn rate 与 checkpoint 示例静态复算一致。未运行章节代码。
- **open_risks**：579 个外链的全量可达性与证据适配复核仍未完成；连续复读当前完成第 01—10 章。下一步从第 11 章继续，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 11 章，2026-10-30）

- **scope**：完整复读第 11 章《长上下文、RAG 与 Agent》，对照第 11 章逐章精修记录，沿“材料—证据—状态—动作—结果”检查 11.1—11.25 的顺序、跨册导航与第 10/12 章交接。
- **continuity_status**：上下文预算与位置利用引入 RAG 的分层链路；证据召回、引用和权限过滤接入 Agent 状态与工具副作用；MCP/A2A、Memory、代码/浏览器 Agent、多 Agent、trace、评估、安全及成本分别落在不同系统责任层，没有把协议、模型策略与执行权限混为一谈。位置槽、RRF 排序、重试序列、路径穿越和条件概率等教学例静态复核一致。第 10 章的平台观测和数据治理之后，本章的外部证据与行动链路顺接；末尾引向第 12 章 reasoning/评估，章间主题推进清楚。第 12 章复读时补齐 11.1 加权上下文分数的非负权重域。5 个代码块保持“静态审阅，未运行”。
- **open_risks**：章节已有精修记录中的 27 个外链尚未完成第四册全量可达性/证据适配终审；整书连续复读当前完成第 01—11 章。下一步继续第 12 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 12 章，2026-10-30）

- **scope**：完整复读第 12 章《Reasoning 与评估》，检查推理状态、候选生成/聚合、验证器、过程监督、搜索、TTC、数学/代码评估、统计比较、benchmark、人评、事实性/校准、安全与发布证据，并对照章节既有精修记录。
- **continuity_status**：章节沿“候选—验证—资源分配—评估”组织，明确区分存在正确候选、验证器选中、业务任务完成；CoT 回归、pass@k、paired lift/bootstrap、切片差异、judge-human agreement、严重度风险和单位正确成本均说明分母、配对/独立假设或 N/A 条件。论文、benchmark、model card 和目标系统实测的证据用途分开。与第 11 章 Agent/RAG 系统评估衔接自然，结尾进入第 13 章安全治理；本章未发现新的正文阻断项，代码未运行。
- **follow_up**：连续复读过程中补充第 11.1 节 `U_context` 的权重约束为每项有限且非负、总和为正，避免将可取负值的加权和称作加权平均。
- **open_risks**：第四册 579 个外链的可达性与证据适配终审仍未完成；整书连续复读当前完成第 01—12 章。下一步继续第 13 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 13 章，2026-10-30）

- **scope**：完整复读第 13 章《安全与治理》，对照章节既有逐章审阅记录，检查从威胁模型、对齐、监督和策略，经过数据流/隐私/评估，到工具、Agent、供应链和治理发布的因果链及前后章节交接。
- **continuity_status**：资产/主体/攻击面/影响定义承接第 12 章评估中的事件和分母口径，并延伸到策略、数据流、隐私、工具权限、事故响应与发布证据；系统安全不被缩成文本拒答。风险排序示例明确是序数启发式；FNR、拒答率、ASR、Wilson 区间、SWR 和群体切片有分母/空样本规则，DP-SGD 示例区分单次教学噪声与正式隐私会计。NIST、OWASP、C2PA、SSDF、SLSA 的类别与结论范围清楚，未将标准、清单或来源签名当作产品认证或系统实测。章末论文索引顺接第 14 章。本轮未改正文，代码未运行。
- **open_risks**：章节外链仍须纳入第四册 579 个外链的整体可达性和证据适配终审；整书连续复读当前完成第 01—13 章。下一步继续第 14 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 14 章，2026-10-30）

- **scope**：完整复读第 14 章《重要论文索引》，检查 14.1—14.46 的索引职责、研究问题摘要、来源链接、证据限定、跨册主讲路由与阅读路径。
- **continuity_status**：本章始终作为代表论文导航而非逐篇课程；实验数字绑定论文模型、数据、配置和预算，机制解释分别回到架构、训练、RAG/Agent、推理、多模态和安全主讲。研究论文、模型报告、数据集、产品文档与协议规范在 14.46 及正文中分开标注。14.42—14.45 将“论文条目是什么”和“方法如何比较”导向第 15 章，作为索引到工程比较的过渡清楚；与第 13 章的安全证据索引也衔接正常。本轮未改正文。
- **open_risks**：本章和第四册其余正文的 579 个外链仍待可达性与证据适配全量终审；整书连续复读当前完成第 01—14 章。下一步继续第 15 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 15 章，2026-10-30）

- **scope**：完整连续复读第 15 章《方法比较与工程取舍》，检查 15.1—15.43 的速查定位、对照逻辑、数值例、公式域、来源边界，以及与第 14 章索引和相邻专题书册的交接。
- **continuity_status**：优化器、Norm、架构、对齐、解码、PEFT、量化、attention/cache、分布式、RAG、Agent、评估与安全条目均先定义共同问题，再比较机制和取舍。Pareto 例、Adam/AdamW 首步、KV 账本、parallel batch、FSDP 理想份额、retriever×reranker 条件链与 pass@k 手算一致；不同论文结果没有合并为统一榜单。章首按条目提供主讲路由，末尾总结把资源交换与失败边界收束为选型方法。本轮未改正文，代码未运行。
- **open_risks**：全册 579 个外链仍待统一终审；整书连续复读当前完成第 01—15 章。下一步继续第 16 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 16 章，2026-10-30）

- **scope**：完整连续复读第 16 章《多模态学习》，沿输入预处理、模态表示、对齐/融合、输出和评估安全链核对 16.0—16.31，并检查与第 15、17、18 章和第十五册主讲的交接。
- **continuity_status**：本章明确区分图像 patch、音频采样/codec、视频时空 token 与文本 token；上下文预算、输入保真、证据召回和 claim 支持分别定位不同失败阶段。CLIP/SigLIP、vision encoder/connector、扩散/视频/语音、统一架构及评估安全各自提供速查入口，完整机制回到专题章节；票据、patch 数、CLIP logits、WER、codec 长度与成本例静态核算一致。章节没有把媒体接口支持等同于任务成功，结尾把部署账本交给第六、十一、十八册。未发现新阻断问题，未修改正文，代码未运行。
- **open_risks**：新增和历史外链仍待第四册 579 个链接的全量终审；连续复读当前完成第 01—16 章。下一步继续第 17 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 17 章，2026-10-30）

- **scope**：完整连续复读第 17 章《图像生成与视频生成》，核对 17.0—17.23 的路线导航、公式/参数约定、可复算例、论文结论边界和向第十五册专题的交接。
- **continuity_status**：从表示与条件进入前向/反向扩散、采样、主干、latent、CFG、自回归生成和图像编辑，再扩展到视频时间状态、world model、FID/FVD、安全与成本账本，概念次序和索引定位一致。噪声/v 参数互换、patch 数与 attention pair、VAE 元素比、CFG 约定、时空 token、FID 示例和采纳成本代理静态复算一致。评估数字绑定数据、模型和论文协议；Sora 被标记为历史研究且说明官方停止服务日期，未作为现行可调用服务。未发现新阻断项，本轮未改正文，代码未运行。
- **open_risks**：章节新增来源与 scheduler 文档仍须进入第四册 579 个外链的可达性/证据适配总审；连续复读当前完成第 01—17 章。下一步继续第 18 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 18 章，2026-10-30）

- **scope**：完整复读第 18 章《语音大模型》，检查波形、声学特征、ASR、codec、TTS、韵律、流式状态、Voice Agent、安全、评估与成本的连续交接，并对照第 17 章和第十五册/第二十一册主讲入口。
- **continuity_status**：采样点、WER/CER 与关键字段、codec ID 数、语音阶段延迟、VAD/endpoint、barge-in 取消以及任务成本均按各自单位说明。Whisper、Qwen3-ASR、mu-bench、Qwen3-TTS 和 LP-TTS-Bench 的数字绑定论文版本、数据和评估配置；端到端音频名称未被当作实时、准确或安全保证。与第 17 章的多模态生成入口分工清晰，结尾归纳内容、时间、权限和证据四条线。本轮补充 `C_success` 的定义域，零成功率时记 N/A；未改代码，示例未运行。
- **open_risks**：外部论文、官方文档和练习锚点仍须纳入第四册整书终审；连续复读当前完成第 01—18 章。下一步继续第 19 章，整册门禁保持未完成，不更新 PDF。

## 第四册整书连续复读（第 19 章，2026-10-30）

- **scope**：完整复读第 19 章《前沿架构与模型更新》，检查机制分类、压缩/索引例、KV cache 数值账本、诊断矩阵、截至 2026-10-30 的官方披露快照及书系专题入口。
- **continuity_status**：序列位置与网络深度明确分轴；递归状态、压缩候选、深度聚合、残差流混合、value 路由和短程 N-gram 容量各自对应不同瓶颈。top-2 漏选和 KV cache 示例静态复算一致，cache 公式的单请求、原始张量与忽略项边界已明示。论文机制、模型卡、发布说明、API 容量和系统卡摘要分别作为不同证据，不用接口字段推断内部架构。与第二十一册 75—95 章及第六、十六、十七册的路由完整；未发现新阻断项，本轮未改正文。
- **open_risks**：第四册正文 579 个外链的可达性与证据适配终审仍未完成；简介、目录与 01—19 章连续复读现已完成。下一步做整册最终门禁，期间不标记本册完成，不更新 PDF。

## 最新工作交接（2026-10-30）

- 第四册第 01—19 章均已完成逐章静态精修；接下来沿简介、目录和主阅读路径连续复读，再做结构、锚点、公式、外链和图示最终门禁。
- 第四册终审后继续全套书余册精修；全套书 goal 仍为 active。用户要求专注修书，不重复核对代理设置，也不更新 PDF。

## 第四册整书门禁检查点（2026-10-30）

- 已从简介与目录开始复核，确认 19 章摘要、分类和第二十一册架构路径一致；第 19 章最新摘要已对应机制轴和来源快照。
- 对简介、目录和 19 章共 21 个 Markdown 文件做本地目标与 GitHub 标题锚点静态扫描：1,257 个链接，678 个本地、579 个外部；本地目标全部存在。依据 [GitHub Markdown 标题锚点规则](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)，修复第 17、18 章各自指向 `EXERCISES.md` 阶段 5 标题的锚点（同一失配出现三处）；复扫后本地文件与锚点均无缺失。
- 这只是整书门禁的静态链接检查；579 个外链的完整可达性/证据适配复核和正文沿主路径连续复读仍未完成。下一步从第 01 章开始连续阅读，并同步检查章节衔接与证据边界；本册仍不得标完成。

## 第四册整书连续复读与最终链接门禁（2026-10-30）

- **continuity_status**：已沿简介、目录顺序完整复读第 01—19 章；第 04 章补清单头/多头注意力复杂度和存储口径，第 09 章修正非法 cleanup 顺序，第 11 章补充加权上下文分数的非负权重域，第 18 章明确零成功率下单位成功成本为 N/A。其余章节连续阅读无新增阻断项。各章简介摘要、目录定位、专题主讲路由和前后章交接保持一致。
- **link_status**：此前 21 个文件的 1,257 个链接静态扫描确认 678 个本地目标和锚点存在，三处练习阶段锚点已修正并复扫。现按去重 URL 检查 349 个外链，346 个 HTTP 200；3 个 HTTP 403 为 OpenAI 官方来源：Sora 停止服务帮助页、GPT-6.1 Terra 发布说明和 Sora 研究报告页。没有 404、5xx 或传输超时。三页在第 17/19 章任务中曾核对其内容；相应证据边界已记入 research。403 表示自动请求未获准，保留为外部可达性复查项，不推断资源不存在。
- **evidence_status**：本轮未新增事实；外链证据适配按各章 `evidence`/`sync_status` 审计记录和对应 research 来源核对。论文结果限定于各自数据、模型、预算和版本；官方接口、标准与模型披露未跨类型外推。三项 403 的正文事实继续保持日期、模型和官方披露范围限定。
- **validation_status**：`git diff --check` 通过；未运行书中代码、测试、模型、benchmark 或 PDF。剩余维护项是三项官方 OpenAI URL 在自动请求中的 403 状态后续复查；第四册主路径复读、链接门禁和精修记录完成，不更新 PDF。全套任务下一步进入第五册。
