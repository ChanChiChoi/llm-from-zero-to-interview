# 第五册按新版写作规范复修（2026-11-17）

## 任务范围

- 目标：按新版《写作规范》重新审视并修订第五册《大模型训练全流程》，重点落实全书读者路径、章节主导任务，以及核心机制在原地使用小例子逐步讲清。
- 范围：简介、目录、18 章正文及受影响的导航/配套资料；不得因追求顺序简洁而删除必要知识、推导、实现、证据或边界。
- 书册契约：读者具备 Python、反向传播、交叉熵和 Transformer 基本数据流基础，不要求多卡训练经验；目标是能够设计、执行、监控、评估和诊断训练工程。
- 阅读路径：第 1—14 章组成训练闭环主线；第 15—18 章是专题路径，其中第 18 章依赖第 17 章的 OPD 基础。
- 编辑尺度：具体技术对象的小例子优先；默认从例子走到术语、规则、必要公式/推导、难点与边界；代码按章节任务选用，若章节承诺实作则必须兑现。上述顺序允许因读者任务、必要定义和数学逻辑调整。
- 研究：凡涉及新增或改变的技术事实，按规范模块 03 定向核验；来源证据写入 `research/`，修改决定和状态只记在本文件。
- 当前状态：已重读新版规范，完成第 1—18 章复修、简介阅读路径补充、四层架构复核、整册路径复读及本地静态门禁。第五册本轮修订完成；外链全量可达性、静态未运行代码和 GitHub/PDF 页面渲染按下方风险如实保留。

## 四层架构复核

| 组织层 | 本轮复核项 | 状态 |
| --- | --- | --- |
| 读者契约 | 读者前置、读完能力、书系分工、必读与选读入口是否说清 | 完成；简介新增本书核心机制的示例—抽象—加深读法，和既有前置、目标、跨册分工一致 |
| 知识路线 | 1—14 章主线及 15—18 章专题路径是否体现依赖、能力递进和交接 | 完成；1—14 训练闭环顺序保留，15—18 专题分支与第 17→18 依赖在简介和目录一致 |
| 章节闭环 | 每章是否从明确问题走到可检查结果，核心机制是否有原地可复走的例子 | 完成；1—18 章均已本轮逐章复读/修改，主线与专题读者路径已复核 |
| 查询与扩展 | 版本/规格/研究争议/跨册深读是否有稳定入口，进阶内容是否没有遮住承诺 | 完成；研究和版本细节留在各章证据/延伸入口，跨册主讲路由与专题先修关系一致 |

## 逐章复核进度

| 章 | 主导读者任务与核心结果 | 复读与例子检查 | 修改与同步 | 状态 |
| --- | --- | --- | --- | --- |
| 01 训练总览 | 从一次训练计划走到账本、checkpoint 与证据链；交付整体训练量估算和异常定位地图 | 全章复读；补齐 PPL、有效 token、token 加权 loss、训练时长、tokenizer、mixture、参数更新、训练阶段、监控、checkpoint 与 mask 的原地例子 | 已在现有段落补例子并串回 1.17 账本；只用了教学假设数值，未改变技术结论和来源；整册主线复读已完成 | 章节复修及整册复读完成 |
| 02 数据工程 | 追踪来源数据如何经治理、筛选、去重、配比和打包进入训练；交付可追踪的数据集与可解释的训练分布 | 全章复读；保留原有 mixture、近重复、研究对照和消融算例，补质量筛选、标签口径、过滤分母、精确/近似去重、污染、DoReMi、SFT mask、偏好与过程标签、多模态对齐、packing、shard、撤回和故障定位例子 | 已在原章节就地补写；确认 EOS 边界标记不等同于 attention mask；仅新增明确标注的教学假设值，未改变研究结论/引用，未改代码；整册主线复读已完成 | 章节复修及整册复读完成 |
| 03 Tokenizer 与数据格式 | 从原文与消息协议走到 token ids、注意力/监督 mask、packing 和可回放回归 | 全章复读；原有成本账本、label shift、packing、扩词表和审计代码已有例子；补最小端到端问答、编码/解码、表示粒度、mask 数组和偏好/验证器小组案例 | 已在对应解释处加入教学样本；合并 §3.10.4 重复的 padding 与 attention 数量复算，并保留有效目标分母、稠密 kernel 可能不节省计算的结论；更正 toy 审计器不实际扩 embedding 的边界；整册主线复读已完成 | 章节复修及整册复读完成 |
| 04 模型架构与初始化 | 从能力/资源约束选模型形状，并把结构映射到参数、训练计算、缓存和稳定性检查 | 全章复读；已保留 RMSNorm、attention/GQA、FFN、MoE、RoPE、初始化及 4.12 完整账本例子；在 SwiGLU 乘法和 MoE 负载项处补局部推演 | 已加入两通道 SwiGLU 门控数值过程、四 token/两个 expert 的辅助负载损失比较；原有论文结论与引用未改，未改代码；整册主线复读已完成 | 章节复修及整册复读完成 |
| 05 预训练目标与训练循环 | 从一个 token 位置追到一次可检查的 optimizer update；交付按有效目标数复算 loss、梯度和恢复状态的能力 | 全章复读；沿同一 [BOS, 猫, 睡, EOS] 样本检查 label shift、teacher forcing、因果可见范围、交叉熵、梯度方向与更新；复核 accumulation、精度、validation、checkpoint 和故障路径中的原有算例 | 将多目标比较移到 5.15 主闭环之后；在 5.0 建立小样本，在 5.1—5.3 接上概率、损失和梯度，说明完整路径并补 BERT、UL2、离散扩散示意例；保留其余内容与论据，章节末三节顺延为 5.16—5.18；未改可执行代码 | 本轮章节复修及整册主路径复读完成 |
| 06 优化器与学习率策略 | 从梯度和历史状态出发，解释优化器与学习率如何组成可恢复、可比较的一次参数更新 | 将主要机制的既有算例前移到术语/公式之前，合并相同输入的重复推演；保留实现差异、状态账本、论文证据与故障边界 | 复读本章；明确 SGD/Momentum、Adam/AdamW、状态内存、Adafactor、Lion、Sophia、Muon、GaLore、参数分组、学习率调度、decay、clipping、batch 与公平比较的例子和因果桥接 | 本轮章节复修及整册主路径复读完成 |
| 07 分布式训练 | 从“一次更新分布到多卡”追踪参数、梯度、激活和 token 的切分/交换；读者能解释通信组、collective、并行布局、恢复状态及主要性能代价 | 全章复读；保留 7.1.2 的 16 卡 DP→PP→TP 层次示例及 rank 组成员；复核有效 token 分母、DDP/ZeRO/FSDP 账本、ring all-reduce、TP 矩阵、PP 时间线、SP/CP、MoE、重算、恢复和诊断 | 在 alpha-beta 模型加入相同 4 MiB、1 条/4 条消息的代入；在 CP 原理处用 8-token 因果注意力跟踪两个 rank 的 Q/K/V；补清 checkpoint 在 optimizer step 边界与累积窗口中途的梯度状态差异；修正重算时间比公式把预期 `>1` 写成定义的做法；去掉 loss 局部均值的重复算式。新增数字均为教学假设，未改变研究来源或 API 结论；未改代码 | 本轮章节复修及整册主路径复读完成 |
| 08 显存优化与数值精度 | 从参数、梯度、状态、激活和临时空间账本定位显存峰值；读者能解释精度格式、重算、分片、低比特和排障分别改变什么 | 全章复读；复核 7B 状态、按生命周期取 peak、长上下文/attention、online softmax、FP16/BF16/FP8/FP4、accumulation、optimizer state、LoRA/QLoRA、OOM 和首个 NaN 的原地算例 | 在峰值账本补两时刻的 live allocation 对照；把 FP16 loss scaling 数值放回格式讲解处；补 FP8 scale/延迟 scale、unscale 后裁剪的短推演；underflow 章节复用前例并保留硬件 subnormal 边界；重申低比特 toy 不代表生产编码或实测 | 本轮章节复修及整册主路径复读完成 |
| 09 训练稳定性与 debug | 从第一次可观察异常出发，交付按时间线收集证据、比较候选根因并验证修复的能力 | 全章复读；复核有效 token loss、异常指示、spike 检测、梯度/相对更新、Adam 预条件、混合精度、分布式 loss 与恢复游标的例子和证据边界 | 在对应公式旁补充 mask 分母、指示量、spike ratio、Adam 二阶矩、clipping、relative update、unscale 顺序和双 rank 加权 loss 的小算例；把练习中的“梯度范数相加”改为标量坐标梯度和 | 本轮章节复修及整册主路径复读完成 |
| 10 Checkpoint、评估与实验管理 | 沿“保存状态—恢复连续—指标可比—实验可追溯—候选可解释”闭环，交付能选择并复核 checkpoint 的能力 | 全章复读；已有保存成本、7B 存储、恢复探针、PPL/stride、domain、benchmark、污染、回归、manifest、公平对照和多目标选择例子保留；检查各机制是否有原地入口 | 增补 FP16/BF16 条件状态集合、较晚 checkpoint 被门槛排除、污染命中率只作候选、run→checkpoint→eval 血缘的具体例子 | 本轮章节复修及整册主路径复读完成 |
| 11 后训练与对齐训练 | 从示范、偏好、程序验证反馈出发，解释后训练数据、更新方法和评估闭环；交付能按任务反馈和资源选择受控实验的能力 | 全章复读；保留 PPO/DPO/SimPO/GRPO/GSPO/DAPO/DrGRPO/DCPO 等方法的推导、算例和论文边界；补齐首次出现时的读者入口并检查重复 | 用同一道算术题串起 SFT、离线偏好和在线 verifier；为 KTO/ORPO 标签形态、RLVR 和安全误拒补小例子；在方法选择处补任务条件到基线方法的逐步决策；合并 GSPO 中重复的长度比率推演 | 本轮章节复修及整册主路径复读完成 |
| 12 多模态训练 | 从一张带票据金额的问题出发，沿视觉表示、图文目标、对话监督到证据评估走完整条多模态训练链；读者能解释多种目标和数据单位的差异 | 全章复读；保留原有 CLIP/SigLIP、VLM、OCR、ASR、视频、Diffusion、Flow Matching、数据、资源、评估算例；补足 CLIP 双向矩阵、VLM shape、CTC 路径和 diffusion 加噪原地推演 | 在对应机制旁新增小型手算；把 caption/VQA 对照例子前移到任务入口；用票据样本重写 15.1 的 VLM 诊断链；删去重复的 caption/计数指标例 | 本轮章节复修及整册主路径复读完成 |
| 13 训练成本与资源规划 | 从参数、token、吞吐和故障状态推到期限与单位成功成本；读者能建立成本账本并按目标集群证据修正纸面计划 | 全章复读；保留 6ND/序列/MoE、active/wall-clock、GPU/MFU、scaling、checkpoint、失败重跑和 60 天计划例子；补成本/成功、MFU 与 HFU 数值入口及诊断小例子 | 在 §1 加固定验收集成本/通过任务例；§5 加 MFU/HFU 代入；§12.3 展示只恢复权重的状态差异，§12.5 展示扩卡效率下降下时间与 GPU-hours 的变化 | 本轮章节复修及整册主路径复读完成 |
| 14 训练系统综合设计与故障诊断 | 将前 13 章机制接成“设计—估算—诊断—验收”闭环；交付审计一份训练方案并区分症状、能力证据与系统成本 | 全章复读；保留 7B 综合账本、DoReMi、spike、能力回归、污染、并行/显存、VLM、偏好和方案审计例；补域 excess loss、加权分数掩盖退化和 batch/scheduler 的局部推演 | 明确章首 `7B/1T/128 GPU` 教学计划；在 §2.2.1、§5.2、§7.2 各补一项小算例；其他系统卡继续链接到机制主讲章，避免重讲 | 本轮章节复修及整册主路径复读完成 |
| 15 领域专家 SFT 与 RL | 从一项专业工作出发，决定语言适配、示范、偏好、可验证反馈和独立权限分别解决什么；交付可追溯的领域训练与发布合同 | 全章复读；保留三条学习链、Fin/Med 研究证据、SFT mask、专家分歧、hard gate、verifier、刷新/遗忘、成本和发布审计算例；补真实任务入口的可手算样本 | 在 §15.0 路线选择后以合成毛利率样本串起 36% 算术、来源版本、SFT、偏好/verifier 和执行权限；增加首读路径提示，允许先走工作流再回读研究证据 | 本轮章节复修及整册主路径复读完成 |
| 16 RLVR 与可验证奖励 | 从任务答案、权限和环境结果定义可验证成功；读者能区分 verifier 的结论与任务真实目标，并沿反馈回路判断能力是否迁移 | 全章复读；保留任务契约、GRPO、测试设计、污染/泛化、成本、demo 和证据边界；检查 verifier 类型、SQL、沙箱和偏好反馈的核心入口是否有原地小例 | 开篇以 `6×7=42` 与 `calculate/write` 两候选演示正确性与权限不能互相抵消，并复用 §16.15 demo；补 schema 业务约束、测试覆盖、等价答案、模型评审、三行 SQL 和沙箱 `permission_denied/unknown` 例；补 RLVR/RLHF 同一客服回答的分工 | 本轮章节复修及整册主路径复读完成 |
| 17 On-Policy Distillation | 从学生自己产生的错误状态出发，让读者解释教师信号如何依赖状态分布、如何进入学生更新，以及如何评估恢复和成本 | 全章复读；保留 exposure bias、sample contract、token/reverse-KL、长轨迹、稳定化、多 rollout、安全、成本、缓存、研究比较和迁移评估；在公式或方法段检查是否能用数值/状态过程复走 | 用“负数金额被 `abs` 错误接纳”作为贯穿状态例；补长轨迹错误累积、token CE、教师监督权重、累计前缀差异、信赖区、教师熵、peer rollout、旧 policy replay、条件恢复率及 tokenizer 对齐的小例；未改变研究结论或代码 | 本轮章节复修及整册主路径复读完成 |
| 18 DeepSeek-V4 系列案例 | 从领域任务、专家 SFT/GRPO 到学生状态上的多教师 OPD，再追到教师调度、运行成本和证据边界 | 全章复读；保留 V4/V4.1-Flash 披露边界、领域专家与内部 routed expert 区别、GRPO、effort/GRM、multi-teacher reverse-KL、调度、路线比较和成本例；检查案例是否先于参数与表格把训练链演出来 | 将开篇生活类比替换为 `17×23` 与负数金额两个技术样本；加首读路线、effort 对照记录步骤、GRM rubric、权限门槛、多教师状态记录、四样本按 teacher index 排序、V4.1 的“问题—环境—验证系统”案例；教学构造均明确不是报告披露 | 本轮章节复修及整册主路径复读完成 |

## 已完成章节的读者路径记录

### 第 1 章：训练总览

- reader_path：从一个 1.3B 参数、120B 有效 token、8 张 GPU 的教学计划出发，先说明一次训练如何依次涉及数据、目标、更新、并行、监控、评估和 checkpoint；再把前文的训练步数、吞吐、FLOPs 与存储检查统一回 §1.17。
- example_status：已在损失、global batch、梯度累积、训练时长、tokenizer、mixture、SGD 更新方向、训练阶段、DPO、监控、checkpoint 选择和零 mask 故障处加入原地算例；数值均为教学假设。算术已手工复核，未把示例当成实测或论文证据。
- formula_status：现有公式保留；新增内容是代入和解释步骤，没有改公式定义。
- code_status：本轮未改代码，原有状态记录不受影响。
- evidence/sync：未改变技术事实和来源口径，无新增研究条目；整册架构复核后确认简介、目录和配套索引无需调整。
- open_risks：整册复读已确认主线路径、章节交接及第 1.17 的账本回收关系；剩余风险是单节信息密度较高，读者可按账本逐项核对。

### 第 2 章：数据工程

- reader_path：从单条网页记录 d-001 出发，跟踪原始字节、解析与规范化、许可、token 化、去重、重复采样和训练曝光；随后用各主题自己的小输入讲清指标、选择和失败边界。
- example_status：新增多标签统计、长度分布、质量筛选排序、PII 与凭证风险区分、过滤率分母、exact dedup 保留选择、Jaccard、污染候选、DoReMi excess loss、SFT assistant mask、偏好标签边界、过程与结果监督、多模态时间戳、packing 利用率、读取游标、许可撤回追踪和中文留存率诊断的原地例子；数值均为教学假设。
- formula_status：公式本身保留；新增计算步骤和对象映射。packing 示例明确区分 EOS 边界标记与阻断注意力的 mask。
- code_status：原有教学审计代码及其状态未改，本轮不重跑。
- evidence/sync：仅增补解释性算例，未新增或改变研究事实、论文结论和引用，无需新增 research 条目；整册复读确认简介、目录、题库及索引无需同步调整。
- open_risks：章节知识密度较高，且 2.19 教学代码横跨多个步骤；整册复读已确认初学者主线清楚。代码仍未运行，读者可先沿 d-001 样本跟完数据处理，再查研究比较和专项主题。

### 第 3 章：Tokenizer 与数据格式

- reader_path：从“用户问 2+2、assistant 回答 4”出发，追踪消息模板如何形成 token ids，哪些 token 进入上下文、哪些成为训练目标，再接到 padding、packing、版本回归和任务 schema。
- example_status：新增假设词表的编码/解码、同一中文串在字符/整词/子词粒度下的示意、三种 mask 的同一组 token 数组、pairwise preference 和组内验证器算例；都明确为教学输入，未指向具体模型 tokenizer。
- coverage/depth：原有词表规模、token/byte、loss shift、packing 边界、媒体 token 数、扩词表参数和回归示例保留；3.10.4 中重复的 padding 数量和 attention pair 计算合并，独有的 6 个有效目标、稀疏 mask 的 kernel 边界仍保留。
- code_status：原有 toy tokenizer 静态审阅状态未改、未运行；补充说明词表增长不等于已实际扩展模型 embedding/输出矩阵。
- evidence/sync：未更改技术结论、研究事实或来源，无新增 research 条目；代码未变，整册复读确认简介、目录和配套入口无需同步调整。
- open_risks：本章内容较长，pack/attention/label 边界仍是高密度段落；整册复读已沿开头样本复走，确认关键符号和前置关系衔接一致。

### 第 4 章：模型架构与初始化

- reader_path：从能力、数据、GPU、上下文和服务约束出发，解释这些条件如何限制 hidden size、层数、head、FFN、位置配置、KV cache 与 MoE；再用参数/训练/运行/稳定四本账筛候选，并以小规模回归验证。
- example_status：原文已有较充分的 local examples（attention shape/因果矩阵、参数估算、训练 FLOPs、RoPE 旋转、RMSNorm、残差尺度、SwiGLU 参数对齐、GQA cache、MoE active 参数/容量、初始化和完整配置账本）；本轮在 SwiGLU 逐元素门控和 MoE aux loss 处补更直接的小算例。
- formula_status：公式与原假设保留；新 MoE 算例明确 f_e/P_e 含义与主任务/容量的局限，新 SwiGLU 算例只演示中间门控输出。
- code_status：本轮未改代码；原文以静态配置与分析流程为主。
- evidence/sync：没有新增或改变研究事实、结果或引用，无新增 research 条目；最终路径复读确认简介和目录摘要仍准确，无需同步调整。
- open_risks：章节术语和账本密度较高，4.12 完整示例的信息量大；整册复读已从变量定义跟到参数、FLOPs、KV 与门槛结论，并核对与第一章训练计划的交接。

### 第 5 章：预训练目标与训练循环

- scope：完整复读并依据新版规范复修 `book-05-llm-training/chapters/05-预训练目标与训练循环.md`。本轮处理读者进入主线的顺序、一个 causal LM 训练样本如何贯穿术语/公式/算例，以及相邻例子的重复。核心技术范围和论文比较均保留。
- reader_path：从 `[BOS, 猫, 睡, EOS]` 出发，先错开输入和标签，再看给定前缀下的序列概率、teacher forcing 和 causal mask；同一位置的三个候选 logits 走到 softmax、交叉熵、logit 梯度和一次简化 SGD 分数更新。随后进入有效 label 分母、三种 mask、packing、整窗更新、梯度累积、混合精度、validation、日志、checkpoint 和排障。读者首次阅读可跳过目标谱系，完成 5.2—5.14 的主闭环后回到 5.15；资料入口、练习和小结现为 5.16—5.18。
- example_status：新贯穿样本的三类 logits `[2,1,0]` 对应目标 `睡`，概率约 0.2447、单位置损失约 1.4076 nat；局部梯度约 `[0.6652,-0.7553,0.0900]`，将分数暂当可调参数、学习率取 0.1 后，目标概率约升至 0.2705、损失降至约 1.3074 nat。这里明确区分直接调整 logits 的教学简化与真实 Transformer 经反向传播更新层参数。原有 2/4-token accumulation、跨 rank 分母、整窗裁剪、loss unscale、token 加权 validation、packing 和恢复算例均保留；其他目标区补 BERT、UL2 和离散扩散的短示意。
- formula_status：补充给定 prefix 下的联合概率分解和单位置交叉熵对 logits 的梯度；数学值已手工/数值复核。原有 loss 分母、MTP 有效目标数、packing 利用率、梯度累积、DDP 平均、裁剪、loss scaling、validation 与 PPL 公式不变。
- chapter_scan/evidence：沿用 2026-10-30 已完成的章级扫描与候选去向，见[第五册既有架构审计](book-05-llm-training/architecture-audit.md)；本轮只重排教学路径并加入明确标记的合成示例，没有引入新的研究事实、论文结论、API 行为或实测数字。既有来源证据仍见 [训练目标研究记录](../research/pretraining-objectives-mtp-and-mdlm-2026.md)、[因果训练与 loss mask 记录](../research/causal-training-and-loss-masks-2026.md)及 [PyTorch 训练 API 边界记录](../research/pytorch-training-api-boundaries-2026.md)。
- code_status：没有改变 5.13 的可执行语句；仍为“静态审阅，未运行（2026-10-26）”，本轮未运行章节代码、模型、benchmark 或 PDF。代码较长且只演示 token-to-logits 表、没有 attention 运算的边界仍在正文说明。
- link/sync_status：已检查 5.15—5.18 标题顺序、围栏配对与本地跨册链接目标（第二册、第二十四册目标均存在）；外链可达性、简介摘要和全册交叉导航留到整册门禁。
- open_risks：整册复读确认主阅读路径、简介摘要和跨章导航一致；内容密度及 5.13 较长代码仍需读者逐步跟读。外链未全量在线核验，代码未运行，GitHub/PDF 呈现未检查。

### 第 6 章：优化器与学习率策略

- scope：按新版规范重新复读 `book-05-llm-training/chapters/06-优化器与学习率策略.md` 并调整讲解路径。章节位置、标题编号、优化器覆盖、研究证据与代码范围保持不变；没有引入新的论文结论、API 行为或实测数据，沿用既有章级扫描及其来源记录。
- reader_path：从单参数的梯度下降出发，先看一次参数和损失怎样变化；再用不同长度样本说明 token 分母，用二维山谷说明标量学习率的限制，继而沿 SGD/Momentum、Adam/AdamW 和不同状态结构看更新方向如何形成。之后用小轨迹讲 schedule、衰减、裁剪、有效 token batch 与公平比较，最后进入代码审计、恢复与故障定位。
- example_status：前移并复用本章已有合成例子：单参数梯度下降；1,000/10 个有效目标的 token mean；二维高低曲率 SGD；Momentum 对梯度反向的延迟；Adam 两步偏置修正；AdamW 与耦合 L2 的二维方向差异；1B 参数状态账本；Adafactor 的 2×3 行列统计；Lion 符号方向；Sophia 曲率预条件；Muon 的矩阵方向；GaLore 的双侧投影；tied embedding/head 分组；warmup 与 cosine/linear 轨迹；WSD/WSD-S checkpoint 分支；累计 decay；global norm clipping；跨 rank token mean；optimizer token-efficiency 与墙钟取舍；Adam 状态恢复；loss-spike 分流。数值为教材构造输入或原研究报告，不写作本地训练实测。
- editing_decisions：把已有算例前置到对应机制首次讲解处，再引出定义、公式、工程边界与证据；没有为了覆盖而给每个术语添加代码或独立例子。合并/删去重复的二维 SGD 推演、Momentum 同一反向轨迹、Adam 两步完整推演、AdamW 同一二维对照、跨 rank token weighting、clipping 首例和公平实验 token/墙钟例；将 warmup、schedule 与恢复算例保留在各自承接位置。保留 Adafactor 近似误差、Sophia 两种曲率估计、Muon 秩亏边界、GaLore 基刷新、分布式 clipping、checkpoint 状态、optimizer 公平调参等非重复的难点和边界。
- formula_status：原公式和推导定义未改；新增/前移的数字均直接代入既有更新式或原有算例，手工检查梯度下降、token 加权、Momentum/Adam 状态、AdamW 两种规则、warmup/cosine、weight decay、裁剪及 token batch 的算术。Dollar 行内公式与代码/公式围栏静态配对检查通过。
- code_status：未改 6.19 的可执行语句、输入输出或代码块状态；继续标记“静态审阅，未运行（2026-10-30）”。本轮不运行代码、测试、训练、benchmark 或 PDF。
- link/sync_status：标题和编号不变，未新增或删除引用链接；简介/目录仍表达相同章节职责，本轮暂不改章节摘要。`git diff --check` 通过；围栏标记为 132 个、行内美元符号数为 1,212，均为偶数。整册复读已确认简介/目录、跨章链接和末端导航与本章职责一致。
- open_risks：章节仍包含较多优化器比较和工程细节；已保留全部核心主题与独立边界，不以篇幅上限压缩。需要在第五册主线路径中检查第六章是否能自然承接第五码的梯度更新，并把优化器状态/裁剪交给第七章并行训练、显存内容交给第八章、恢复与评估内容交给第十章。

### 第 7 章：分布式训练

- scope：按新版规范重新通读 `book-05-llm-training/chapters/07-分布式训练.md`，检查整章读者路径及核心机制是否在原地用可复走的例子讲解；保留章节编号、并行方法覆盖、论文证据和原有资源账本代码。
- reader_path：从四张卡处理不同数据开始，先说明一次更新为何需要同步；用 16 卡布局回答“每张卡保存什么、哪些卡合作算一层、stage 如何前后相连”，再沿有效 loss 分母、collective、状态分片、层内/层间/序列切分，推进到 MoE、通信重叠、checkpoint、故障与性能诊断。7.1.2 的布局层次已按“16 卡分 4 份数据布局—每份 2 个 PP stage—每 stage 2 张 TP 卡”写清，本轮确认保留。
- example_status：7.2.2 手算 100/200/400 个有效目标与局部均值，删除重复展示；7.4.6 保留 4-rank、4-element 向量从 reduce-scatter 到 all-gather 的消息轨迹，复核每 rank 的 12 MiB ring 发送量；7.4.7 新增 alpha=0.25 ms、β=1、4 GiB/s 下同传 4 MiB 时一条大消息约 1.23 ms、四条串行小消息约 1.98 ms 的教学代入；7.8.2 新增 8-token、CP=2 的因果可见范围表，展示 rank 1 为什么需要 rank 0 的 K/V；7.11.2 以 100/130 ms 解释重算时间比 1.3；7.13.1 以 step 边界和 4-step 累积到一半对照是否需要保存未完成梯度。原有 ZeRO、TP、PP、MoE capacity、DCP 重分片、MFU/HFU 等示例保留。
- formula_status：除局部均值重复算式删除外，公式定义保留；`R_compute` 改为实测比值，`>1` 写作相同输入/计时范围下的观测解释，不再当作定义条件。手工核对 alpha-beta 例子的单位换算、CP 因果位置范围、重算时间比和已保留算例。
- code_status：7.17 Python 资源账本代码未改，仍标记“静态审阅，未运行（2026-10-30）”；未运行章节代码、测试、模型、benchmark 或 PDF。
- evidence/link/sync_status：本轮修改只加入明确标注的教学输入、澄清测量口径和复用既有技术结论；未引入新研究事实、来源或链接。整册复读确认章节结构、简介职责和跨章导航一致，无需修改目录或配套索引；外链未全量在线核验。
- open_risks：ring all-reduce 例子经手工核对，未在真实 NCCL 拓扑上运行；CP 例子用于解释因果可见范围，不代表所有框架采用相同 rank 排列和通信协议。综合账本代码未运行；整册复读已检查主路径和跨章锚点，外链未全量在线核验，GitHub/PDF 渲染未检查。

### 第 8 章：显存优化与数值精度

- scope：依据新版规范重新通读 `book-05-llm-training/chapters/08-显存优化与数值精度.md`，检查从显存账本到精度选择、OOM/NaN 诊断的主路径；保留 8.0—8.19 结构、研究比较、版本边界与现有 Python 教学账本。
- reader_path：先区分逻辑状态、活张量和设备分配，再按峰值时刻理解权重、激活、梯度、optimizer state 和 workspace 的共存；随后沿激活与 attention 规模、数值表示/scale、checkpointing、accumulation、sharding/offload、LoRA/QLoRA 展开，最后按 phase 定位 OOM 与第一处非有限值。
- example_status：在 8.1.5 新增第一次 optimizer step 的两时刻假设账本：forward 22 GiB、optimizer step 26 GiB，说明 peak 取时间最大值而非分项峰值相加；8.6.3 把 $2\times10^{-8}$ 梯度、scale 4096、scaled value $8.192\times10^{-5}$ 的算例前移；8.6.5 新增简化 scale-round-dequantize 例及 delayed scaling 遇到新 amax 的 4→8、E4M3 max 448 饱和演示，并标明不是实际 FP8 kernel 公式；8.7.3 用 $(6,8)$、scale 8、阈值 5 对照正确 unscale/clip 与错误顺序，得到 $(3,4)$ 和 $(0.375,0.5)$；8.13.3 引用前例并聚焦 subnormal/flush-to-zero 的实现边界。算例均为教学输入。
- formula_status：仅在对应公式旁补数值演算；量化 example 说明采用等间距码值作抽象，不冒充 E4M3/E5M2 精确表示；手工核对 peak、scale、裁剪、loss scaling 的算术。原有权重/状态、attention、LoRA/QLoRA 和研究公式未改。
- code_status：8.15 唯一 Python 教学代码的可执行语句未改，沿用 2026-10-26 的“已验证”正文状态；维护记录仍说明解释器小版本缺失。本轮未运行代码、测试、GPU、模型、benchmark 或 PDF。
- evidence/link/sync_status：复用既有 2026-11-02 章级扫描及 `research/activation-memory-and-low-precision-training-2026.md` 的证据，不增加研究事实、来源或链接。整册复读确认章节标题、结构、简介职责和配套索引一致；外链未全量在线核验，版本入口沿用原有证据范围。
- open_risks：峰值账本、FP8 scale 与 clipping 数字均为纸面教学输入，不代表设备测量；E4M3 max 的格式事实仍按既有来源范围使用。代码既有验证没有记录 Python 小版本；整册复读已确认跨章路径，PDF 和 GitHub 公式渲染未检查。

### 第 9 章：训练稳定性与 debug

- scope：在既有第九章审阅和 2026-11-02 章级扫描基础上，按新版规范完整复读 `book-05-llm-training/chapters/09-训练稳定性与debug.md`，重点检查公式附近是否有读者能复算的输入、动作和结果；保留 9.0—9.16 结构、诊断范围、研究比较及已有综合 toy，不删核心主题。
- reader_path：沿“健康基线—第一次异常—候选机制—固定现场对照—修复回归”阅读。读者先看 loss 和监督分母怎样对应，再把 step 标记为待查对象；之后分别检查 spike、坏值、梯度/更新、数据 mask、optimizer/精度、rank 同步和恢复状态；最后用综合时间线串起症状、现场证据与 checkpoint 完整性。
- example_status：9.2.1 用四个位置 loss [0.2,0.8,1.4,9.0] 和 mask [1,0,1,0] 得到有效 token 数 2、正确 mean 0.8，对照误把四格都平均所得 2.85；9.2.2 用有限 loss/梯度、64 个有效 token、spike ratio 2.4、阈值 2.0、rank spread 0.1 展示指示量为 1 只会触发留现场；9.4.2 用健康 loss [2.40,2.18,2.05] 和当前 5.80 得到 ratio 2.66；9.4.5 用 beta2=0.9、v=1、梯度 0.1→1 演示未偏差修正二阶矩 0.901→0.9109 和简化预条件量变化，并明确不证明实际 spike 根因；9.6.1 用 (6,8)、范数 10、阈值 5 得到 (3,4)；9.6.2 用参数 (3,4) 和更新 (-0.03,0.04) 算出 relative update 1%；9.9.3 用 scale=8 对照正确 unscale 后 clipping (3,4) 与错误顺序所得 (0.375,0.5)；9.10.3 用两 rank 的 2/6 个有效 token 和 3/1 的 local mean 得到全局 loss 1.5、等权 rank mean 2.0。9.15 练习改为标量参数坐标的未归一化梯度和，避免把向量范数误当成可加权的梯度值。新增值均为教学输入。
- formula_status：本轮不改目标定义和研究公式；手工复核新增分母、spike ratio、二阶矩、范数裁剪、相对更新、loss unscale 与 rank 加权算术。新加入的行内美元分隔按行配对，章节公式围栏仍为 62 个围栏标记、31 对。
- chapter_scan/evidence：复用既有 2026-11-02 第九章扫描及 [training-stability-and-debug-2026](../research/training-stability-and-debug-2026.md) 与 [initialization-and-training-stability-2026](../research/initialization-and-training-stability-2026.md) 的证据边界。本轮只补教学数值、修正练习表达，没有新增论文事实、API 行为或生产测量，不增加 research 内容。
- code_status：9.12 Python 诊断示例未修改，仍标记“静态审阅，未运行（2026-11-02）”；本轮未运行代码、测试、训练、模型、benchmark 或 PDF。
- link/sync_status：章节编号、标题和链接目标均未变；正文没有新增链接，无需改简介、目录或知识配套。`git diff --check` 通过；62 个围栏标记配成 31 对，所有新增行内美元符号成对，无意外控制字符。
- open_risks：数值例子只演示口径与机制，不代表真实训练测量；Adam 二阶矩例子省略偏差修正、一阶矩与 epsilon 在更新式中的作用，已在正文标明。9.12 示例仍未运行；整册复读已确认主线路径和跨章链接，GitHub/PDF 渲染未检查。

### 第 10 章：Checkpoint、评估与实验管理

- scope：依据新版规范完整复读 book-05-llm-training/chapters/10-checkpoint评估与实验管理.md，检查从训练状态保存到 checkpoint 选择的读者闭环；沿用既有 2026-11-02 章级扫描和来源，不改 1—19 章节结构，不删评估合同、统计、污染审计、复现与选择方法。
- reader_path：先确定一次训练要保存哪些状态，再算保存间隔与容量；通过中断前后的一步探针验证恢复；随后逐步解释 validation/PPL、stride、domain loss、benchmark 协议和统计波动、污染信号、回归率、实验血缘、公平对照及候选门槛；最后把这些证据回收到 checkpoint 选择和可复盘记录。
- example_status：保留原有 2M token/500 step/1B 最大回退和 6% 串行保存成本、7B 权重 13.04 GiB/恢复状态预算、恢复探针偏差、两 token PPL=2.83、context=4/stride=2 的唯一目标计数、domain 加权 loss 2.2、配对准确率变化、回归率 5%、记录字段覆盖 66.7%、104.9B token 学习率对照、复现容差和第 14 节候选加权排序。新补例子：FP16 checkpoint 缺 scaler 而 BF16 不把 scaler 视为必需字段；ckpt_3000 的 val loss 虽为 2.16，但 code loss 2.98、safety 0.89、11 个回归失败且不稳定，按 0.92/5 门槛排除；100 道评测题中 3 个短 n-gram 命中只得到 3% 待核查候选率，不代表记忆污染比例；run_042 → ckpt_2000 → eval_017 展示指标与原始结果的追溯链。新增例子均为教学输入或记录示例。
- formula_status：保留状态集合、保存成本、恢复偏差、PPL/stride、domain 聚合、回归率、统计检验、实验覆盖率、多变量差异和归一化选择公式；本轮仅添加状态集合的条件例，未改定义和推导。
- chapter_scan/evidence：复用既有 2026-11-02 第十章扫描及 [checkpoint-evaluation-and-experiment-management-2026](../research/checkpoint-evaluation-and-experiment-management-2026.md) 的来源边界；本轮未增加研究事实、论文结论、API 行为、实测数字或来源，不新增 research 内容。
- code_status：15 节标准库 demo 未改，原“已验证（2026-11-02，Python 3.12.9，标准库）”状态和执行记录继续适用；本轮未运行代码、测试、DCP、多 GPU、对象存储、benchmark 或 PDF。
- link/sync_status：章节结构、标题和链接目标未变，没有新增链接；简介/目录、练习和索引无需因本轮例子改动而同步。章节共有 86 个代码围栏标记、43 对；行内美元分隔逐行配对；git diff --check 通过。
- open_risks：PyTorch DCP 2.15 和动态 Transformers 文档仍按既有版本边界解释；外链未全量在线核验，GitHub/PDF 公式渲染未检查；整册复读已确认交叉导航，代码输入边界仍按正文中的静态审阅范围理解。demo 本轮未变且未重跑。

### 第 11 章：后训练与对齐训练

- scope：依据新版规范完整复读 book-05-llm-training/chapters/11-后训练与对齐训练.md，保留 1—20 结构、全部主要方法、证据和算法边界；在已有充分算例处不重复堆例子，针对方法谱系入口和标签形态补足具体对象演示。
- reader_path：先以同一道可验证算术题区分 SFT 示范、chosen/rejected 离线比较和当前 policy rollout 的程序反馈；随后走过 SFT/RM/PPO、离线偏好方法、GRPO 系列长推理更新、公开模型披露、方法选择、数据/安全风险及综合诊断 demo。
- example_status：保留并复读原有 SFT token mask、RM/B-T 概率、PPO ratio 与 GAE、DPO margin/梯度、SimPO 长短 response mean、GRPO group advantage/零方差、GSPO ratio/clipping/reduction、DAPO clip/采样/删失、DCPO bounds/RUR 分母、R1 数据比例、OPD KL、选择比较、风险分母及综合 Python trace。新增：开篇用 3/4+1/4 同题说明示范、固定 preference pair 与在线 verifier 的反馈差异；1.4/1.5 复用该例解释离线数据不能评新采样及 verifier parser 的作用；1.6 以危险请求正确拒答、密码重置请求过度拒答区分安全召回与 over-refusal；4.1 用 chosen=4/rejected=5 与单条 desirable 标签区分 DPO/IPO 和 KTO 数据形态；13 节将无 verifier 的聊天偏好数据、可执行代码测试和各自资源条件映射到 DPO/GRPO 基线及后续对照。新增场景为教学示例。
- editing_decisions：以技术小样本替换开篇驾驶类比，让读者直接看见反馈数据如何不同；保留 GSPO 的双 token、长序列、逐 token/sequence clipping 和外层 reduction 例子，只删除与此前 1.2^n 对照重复的一段数字复算，改由 1,000-token log-ratio 例承接统计含义。
- formula_status：未改算法定义、论文公式或数值结论；GSPO 重复内容压缩不影响其“几何均值 surrogate 非标准整句概率比”的边界。新增数值为简单合成样例，无训练测量含义。
- chapter_scan/evidence：复用既有 2026-11-02 第十一章扫描以及 research/rlhf-preference-and-policy-optimization-2026.md、research/llm-training-and-post-training-systems-2026.md 的版本和来源边界；未新增技术事实、论文结论、API 行为、实测数据或来源，不改 research。
- code_status：第 17 节唯一 Python 分析 demo 的可执行语句、输入输出、依赖、预期和状态均未改；此前 Python 3.12.9 标准库主路径验证继续适用，异常分支静态审阅状态不变，本轮未运行代码或测试。
- link/sync_status：章节编号、标题和链接目标不变；简介、目录和配套索引无需因教学示例调整而同步。全章代码围栏标记 134 个、67 对；行内美元分隔逐行配对，无意外控制字符；git diff --check 通过。
- open_risks：方法覆盖和证据密度仍高，尤其 GSPO/DAPO/DCPO 比较；整册复读确认首读路径和跨章链接清楚，方法表仍适合作为后续查阅入口。早期预印本及产品披露沿用原证据限定；外链未全量在线核验，GitHub/PDF 渲染未检查。

### 第 12 章：多模态训练

- scope：按新版规范完整复读 `book-05-llm-training/chapters/12-多模态训练.md` 的 18 节；保留图文对比、VLM 桥接、instruction tuning、caption/VQA、OCR/文档、ASR/语音生成、视频、diffusion/Flow Matching、多模态数据、对齐、资源、安全与综合 demo 的范围和原有研究边界。章节已有大量具体算例，本轮只在核心机制首次出现处补缺，不按“每节一个例子”凑配额。
- reader_path：从“读取票据金额”及模型是否使用媒体证据的问题开始；由全局图文匹配进入 VLM 中视觉 token 与文本 token 如何接合，再看 assistant-only 监督、局部 OCR/VQA、语音对齐、视频时间关系和连续生成目标，最后用数据协议、资源账本和分层评估验收训练结果。
- example_status：保留原有 SigLIP 假负例与归约、tokenizer/mask、视觉 token budget、分阶段训练、caption/VQA 聚合、OCR/CER/字段关系、ASR 帧与错误传播、codec code 数、视频 token/tIoU/时间偏移、diffusion Min-SNR、Flow Matching、样本过滤、跨模态反事实、IoU、资源峰值趋势和 unsupported-rate 等例子。新增：§2.1 以两对图文的 `[[3,1],[0,2]]` 相似度矩阵计算 i2t loss `0.127`、t2i loss `0.181`、双向平均 `0.154`；§3 用 `2×2` 视觉特征经 `2×3` 投影拼成 `[5,3]` 的 VLM 输入；§5 将三杯与局部计数例前移到 caption/VQA 入口并删除重复的 20+20 指标复算；§7 用两帧 CTC 路径 `0.36+0.24+0.24=0.84` 得 loss `0.174`；§9 用同一 `x0=0.8, ε=-0.4` 与三个 `ᾱ` 值复算前向加噪，并连接到后文 demo；§15.1 用金额由 128 改为 182 的同模板票据串起媒体、connector、mask、冻结 LLM 的梯度和反事实诊断。新增均为明确的教学样本。
- formula_status：没有改变目标定义、论文公式、方法结论或研究数值；新增数值手工复算。保留原有公式及定义域边界。新式样 CLIP 矩阵与 CTC 路径均接在公式之前/旁作小例子，不替代一般定义。
- chapter_scan/evidence：复用 `progress/book-05-llm-training/architecture-audit.md` 中 2026-11-02 对图文目标、VLM/课程、SFT 合同、语音、视频、连续生成及相关论文/报告的定向扫描和候选去向；本轮未增加技术事实、引用、模型/API 行为、论文结论或实测数据，不新增 `research/` 记录。
- code_status：§12.1 约 300 行标准库 demo 未修改，正文状态仍为“静态审阅，未运行（2026-11-02）”；未运行章节代码、模型、benchmark、测试或 PDF。
- visual/link/sync_status：保留既有流程、比较表、数据轨迹与算例；未新增 Mermaid、图像或链接。标题和编号未变，不触发简介、目录及题库索引同步。`git diff --check` 通过；章节围栏标记 106 个、53 对，行内美元分隔逐行配对，无目标下标 `x_{<t}`/`y_{<u}` 命中及意外控制字符。章节原有 54 个外链的在线可达性未复查。
- open_risks：本章整体较长且跨多个模态；整册复读已检查初学者阅读节奏、与第 15 章专题分工及 12.1 demo 的位置。Qwen3-VL、Qwen3-Omni、Qwen3.5-Omni、CogVideoX 等配方和数字沿用原报告限定，本轮未重新核源；该章 54 个外链未逐一在线核验，GitHub/PDF 数学渲染未检查。

### 第 13 章：训练成本与资源规划

- scope：完整复读 `book-05-llm-training/chapters/13-训练成本与资源规划.md`，保留 15 节成本模型与规划闭环、既有论文比较、全部公式和唯一 Python 教学 demo；本轮仅为未在原地算开的成功任务成本、MFU/HFU 和诊断路径补例子，没有改变论文事实、资源基线或章节结构。
- reader_path：从“7B 需要 128 张卡还缺哪些条件”开始，建立计算/存储/网络/数据/人力/失败的成本账本；再把参数和有效 token 转成 FLOPs、把目标卡组吞吐转成主循环/墙钟时间，以序列长度、MoE、扩展效率、MFU/HFU说明纸面值与实测差异；后续通过训练 token 与生命周期推理、checkpoint 容量/带宽、失败窗口和 60 天资源计划决定预算；最后用诊断问题检查单位、状态和扩卡假设。
- example_status：保留既有电力与碳排、`6ND`、长短序列、MoE 激活/总参数、吞吐与 availability、PaLM/MegaScale、128 卡 deadline、Chinchilla/lifecycle、7B checkpoint、保存间隔和 60 天计划算例。新增：§1 以 1,000 GPU-hours/820 次通过与 900 GPU-hours/600 次通过的固定验收例计算 1.22/1.50 GPU-hours per pass，并说明 80% 质量门槛先于单位成本；§5 用 7B、200k tokens/s、128 张假设卡的 FLOPs 代入 MFU 6.6%，再以额外 20% 重计算示意 HFU 约 7.9%；§12.3 用 step 1,000 只保存权重导致 Adam 状态和 data cursor 重置说明“能加载”不等于“能续训”；§12.5 沿用 128 卡例并假设 256 卡 `eta=0.40`，吞吐 409.6k token/s、固定 token 时间为原来的 68.75%、GPU-hours 增约 37.5%。以上新增均为教学数值，非硬件/生产测量。
- formula_status：无算法定义或论文公式变更。新增单位成功成本公式要求成功数大于零且成本计量单位一致；新增 MFU/HFU、扩卡时间和卡时均手工复核，分别为 6.6%、7.9%、68.75% 和 +37.5%。
- chapter_scan/evidence：复用 `progress/book-05-llm-training/architecture-audit.md` 中 2026-11-02 章级扫描及 `research/training-cost-and-resource-planning-2026.md` 的来源边界；本轮只增教学代入，没有增加研究事实、价格、设备性能或来源，因此未改 `research/`。
- code_status：§10.1 Python 标准库规划 demo 的可执行语句、状态、输入和输出均未改，沿用“静态审阅，未运行（2026-11-02）”及手算输出说明；本轮未运行代码或测试。
- link/sync_status：未改标题、结构和链接，没有触发简介、目录、练习或配套索引同步。`git diff --check` 通过；章节围栏 84 个、42 对，行内美元分隔逐行配对，无控制字符。
- open_risks：所有计算/成本例仍是教学假设；单价和目标集群长期吞吐需要项目现场数据。§10.1 demo 中 `recovery_assumption_recorded` 的检查范围已在正文说明，但 demo 未运行。整册复读已确认主线路径和跨章链接；PDF/GitHub 呈现未检查。

### 第 14 章：训练系统综合设计与故障诊断

- scope：完整复读 `book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md`，保留 20 节的综合设计、spike/回归/污染诊断、跨章复核卡和方案审计 demo；继续遵守本章不取代第 2—13 章机制主讲的边界。
- reader_path：以 `7B`、`1T` 有效 token、`128` 张 GPU 的预训练计划为统一对象，先区分目标、分母和证据，再沿 data mixture、tokenizer、架构、优化、并行、状态与评估形成账本；之后由首次 spike 追到反事实复现、能力回归和污染覆盖率，最后连接 batch/scheduler、跨章配置复核、偏好方法选择与完整训练方案验收。
- example_status：保留原有 7B/1T/128GPU 方案账本、DoReMi 研究案例、logit 共同平移与 BF16 量化、Adam 二阶矩、领域 loss、污染 coverage、global batch、训练状态、VLM 反事实及 §15 完整账本。新增：§2.2.1 两个域的 reference loss `[2.0,3.0]` 与 proxy loss `[2.2,4.0]`，得到 excess `[0.2,1.0]`，只说明 Group DRO 的权重压力方向，不伪装成最终权重公式；§5.2 在 900 道通用题和 100 道代码题上，旧版本 `.90/.80` 得总分 `.89`，新版本 `.95/.30` 得 `.885`，展示总分近似不变而代码退化 50 个百分点；§7.2 以 1,000,000 个有效 token、1k/4k token per update 得 1,000/250 updates，说明固定在 step 800 衰减会导致两个实验的 token 时间轴不同。新增数值均为教学样例。
- formula_status：未改公式定义、研究数字、结果或章节间的机制边界；新增运算已手工复算：DoReMi excess `.2/1.0`，分层总分 `.89/.885`，训练更新 `1000/250`。
- chapter_scan/evidence：复用 `progress/book-05-llm-training/architecture-audit.md` 2026-11-02 章级扫描，以及原文对 DoReMi、Molybog、Bai、PALOMA、PaCoST、TierCheck、DCP 2.15 和有限精度论文的来源限定；本轮没有新事实、版本/API 断言或研究来源，不新增 `research/` 记录。
- code_status：§15.1 Python 标准库账本 demo 未改；仍为“静态审阅，未运行（2026-11-02）”，没有运行章节代码、测试或系统实验。
- link/sync_status：章首和小节标题未改；本地链接、外链、目录和配套索引不变。静态检查 `git diff --check` 通过；章节围栏 56 个、28 对；行内美元公式逐行配对，无禁用尖括号下标和控制字符。
- open_risks：整册复读已检查 §§7—14 配置复核卡与机制主讲章的分工，未发现需要重排的重复；§15 demo 未运行，GitHub/PDF 公式呈现未检查，外链未全量在线核验。

### 第 15 章：领域专家 SFT 与 RL

- scope：完整复读 `book-05-llm-training/chapters/15-领域专家sft与rl.md`，保留 30 个编号小节和原有三条连续主线：领域缺口到训练信号、工作契约到目标和评估、专家分歧到发布审计。本轮只在选择训练路线的入口补小样本并标明研究细节的首读位置，没有改变覆盖、研究结论或章册分工。
- reader_path：由“专业工作如何算完成”引入；沿一份财报样本从术语/数据缺口，走到字段化 SFT、合规答案偏好、可算奖励/verifier，再区分证据来源、动作权限和发布门槛；随后沿工作契约、样本 schema、分布、专业正确性、专家分歧、hard gate、时间刷新、成本和审计记录完成闭环。
- example_status：保留原有代表研究对照、金融毛利率 `(100−64)/100=36%` 与可比期间降至 31% 的手算、医疗摘要三检查、数据桶加权 `.905/.725`、严重性加权风险、token mean/sample mean `.68/.8`、hard release conditions、source revision、成本 `315/42=7.5`、候选发布记录与异常分支。新增 §15.0 的金融入口小例：`Revenue=100`、`COGS=64` 得 `36%`，并说明数值正确仍要满足主体/期间/版本、呈现协议、可验证算术和独立动作授权；同时将 §15.0.1—15.0.2 的论文/评测表标为证据阅读路径，首读者可先沿 §15.1—§15.9 的工作流和实例。
- formula_status：无公式或研究数值改动；新增毛利率手算为 36%，与 §15.6 后续算例一致。
- chapter_scan/evidence：复用既有 `progress/book-05-llm-training/architecture-audit.md` 中 2026-11-02 扫描及 FINDAP、ICD 生成式编码、FinGPT、TIMER、MedAlign、FinBen、DeepSeekMath 等来源边界；未增加研究事实、引用、API 行为或论文解释，因此不改 `research/`。
- code_status：§15.28 Python 标准库发布审计 demo 未改，仍为“静态审阅，未运行（2026-11-02）”；本轮未执行章节代码或测试。
- link/sync_status：章节标题、编号、链接、练习和配套索引未变，无同步触发。`git diff --check` 通过；三反引号围栏 18 个（9 对），波浪围栏 14 个（7 对），行内美元分隔逐行配对，无控制字符。
- open_risks：§15.0.1—15.0.2 研究表仍信息密集，已提供先读工作流的路径；整册复读已确认读者可从第 12 章自然转入领域任务，再进入第 16 章 verifier 专题。预印本证据沿用原作者/配置限定，§15.28 demo 未运行，外链未全量在线核验，PDF/GitHub 渲染未检查。

### 第 16 章：RLVR 与可验证奖励

- scope：完整复读 `book-05-llm-training/chapters/16-rlvr与可验证奖励.md`，保留任务边界、成功契约、verifier 质量、GRPO、奖励投机、污染、迁移评估、成本、可重放 demo、安全约束和方法选择；主要改动是在初学者首次遇到关键判定时增加具体对象和可复走过程，没有改变章节覆盖、研究结论或算法定义。
- reader_path：由同一道 `6×7=42` 任务说明结果正确与动作授权是两项独立条件；沿规则、程序、等价性、专家/模型 verifier 看各自漏检点，再进入 verifier 质量和 GRPO 更新；最后用独立测试、污染审计、沙箱记录、成本与 RLVR/RLHF 例子判断 reward 增长能支持多强的能力结论。
- example_status：保留 soundness/completeness 分母、三条 GRPO 组内优势、PPO clip、sequence/token reduction、reward hacking、holdout 污染率、transfer gap、单位成功成本及 §16.15 JSON verifier。新增开篇正确答案与 `calculate/write` 动作对照；§16.4 的 schema 业务范围、测试遗漏负数、等价数值和 unsupported root-cause 摘要；§16.6 三行 `users` 数据的 SQL 解析/执行差异；§16.7 `permission_denied` 与 `verifier_error/unknown` 轨迹；§16.18 同一客服事实由结果规则和偏好分别评价。新增均为教学构造。
- formula_status：未改研究公式或算法定义；新增计算仅为整数/等价值/有限条件演示，无新的量化主张。
- chapter_scan/evidence：复用既有 `progress/book-05-llm-training/architecture-audit.md` 中 2026-11-02 第十六章扫描和 `research/verifier-quality-and-reward-hacking-2026.md`、`research/rlhf-preference-and-policy-optimization-2026.md`、`research/test-time-compute-and-evaluation-2026.md` 的证据范围。本轮没有新增论文结论、接口行为、实测数字或来源，不增加 research 条目。
- code_status：§16.15 Python 标准库 demo 未改，保留“静态审阅，未运行（2026-11-02）”；未运行章节代码、测试或外部工具。
- link/sync_status：标题、编号、链接、练习和配套索引未变；整册复读确认简介摘要仍准确；本地静态门禁结果已回填。
- open_risks：研究对照表仍密集，首次阅读应沿任务契约和可复走例子进入；论文数字仍受原任务/模型/harness 限制，demo 未执行，外链未全量在线核验，GitHub/PDF 渲染未检查。

### 第 17 章：On-Policy Distillation

- scope：完整复读 `book-05-llm-training/chapters/17-on-policy-distillation.md`，保留离线/在线分布差异、状态 schema、token 和 sequence 目标、教师可靠性、长轨迹、稳定化、peer rollout、安全、成本、缓存、研究对照和能力评估。本轮按新版规范把抽象概念接到学生真实生成状态，未删减算法或研究覆盖。
- reader_path：以金额解析器错误地把 `-3` 取绝对值作为贯穿状态，先说明离线示范看不到学生错误前缀；再把该状态映射到训练单元字段，跟随教师分布和 reverse-KL/策略梯度目标，随后检查长轨迹差异累积、教师不确定性、同题成功/失败 rollout，最后用旧轨迹混合、条件恢复率、能力迁移和 tokenizer 对齐检查系统是否真的学会恢复。
- example_status：保留原有概率错误累积公式、reverse-KL 两动作手算、教学代码 agent 状态链、教师调用成本、sampled estimator、完整研究案例和配对 20 题评估。新增：每步错误率 `.1` 下 3/10 步至少一次错误概率 `27.1%/65.1%`；`p_T=(.8,.2)`、`p_S=(.5,.5)->(.7,.3)` 的交叉熵 `.693->.526`（`T=1`）；教师权重示意 `.8×1×.5=.4`；累计 KL `0,.05,.15,.45` 与教学阈值 `.2`；teacher/student 单 token 概率 `.8/.02` 的 log-ratio `-3.69`；教师熵 `.693/.199`；三条同题结果区分答案错与权限错；revision 12/11 混合 `80%/20%`；等难随机分配失败状态的 `60%/40%` 恢复率；简化 tokenizer 切分差异。新增数值均为教学假设。
- formula_status：未改变方法定义或论文结果。新增算术已手工复核；示意权重、独立错误假设、信赖阈值和恢复率均明确标为教学输入，不可外推为训练结论。
- chapter_scan/evidence：复用既有 2026-11-02 第十七章扫描和 `research/llm-training-and-post-training-systems-2026.md` 的论文、报告及 API 版本边界；本章 2026-11 新增的 Bernoulli OPD、Learning Dynamics、BOPD 三项研究对应独立证据记录 `research/opd-papers-2026-11.md`，其中明确标注预印本作者结果、模型/任务范围和未验证边界。本轮只补教学例，没有新增技术事实或来源，不改 research。
- code_status：§17.16 教师调用选择 Python demo 未改，保留“静态审阅，未运行（2026-11-02，Python 标准库）”；本轮未运行代码或测试。
- link/sync_status：标题、编号、链接、练习和配套索引未变；整册复读确认简介摘要仍准确；本地静态门禁结果已回填。
- open_risks：研究比较和框架文档部分仍密集，读者可先掌握状态、损失、例子再查方法表；外部预印本/动态 API 边界不变，demo 未运行，GitHub/PDF 渲染未检查，外链未全量在线核验。

### 第 18 章：DeepSeek-V4 系列后训练案例

- scope：完整复读 `book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md`，保留报告版本与披露边界、领域专家与内部 MoE routed expert 的区别、SFT/GRPO、effort 与 GRM、多教师 full-vocabulary OPD、teacher scheduling、V4.1-Flash 执行管线、方法比较和成本判断。本轮集中用具体技术对象替代开篇日常故事，并补充抽象披露的过程例。
- reader_path：先用 `17×23=391` 与“负数金额 `-3` 必须拒绝”两项教学任务看专家培养，再沿学生生成的错误状态看多教师目标；之后复算 reverse-KL 和几何平均，追踪 logits 内存账本与按教师分组调度，最后对照不同合并路线、单位成本和 V4.1 的“问题—环境—验证系统”。四步首读提示把主线和研究查阅内容分开。
- example_status：保留 GRPO 奖励 `[.2,.8,.5,.1]`、三 token reverse-KL/单样本方差、两教师几何平均、`8192×100000×2` logits 缓冲、full-route/按需路由成本和 1,000 请求 break-even。新增相同解析任务的 SFT/RL/OPD 教学链；effort 对照记录字段（不预设未披露模式结果）；synthetic GRM rubric `[1,0,1]`；功能正确但权限失败的乘法门槛；学生状态契约实例；四条 `数学/代码` 样本按 teacher index 分组并保持 loss denominator；V4.1 合成任务环境与验证器。新增内容明确不是 V4/V4.1 内部数据、配置或实验结果。
- formula_status：未改变原技术报告公式、数值或论文事实；只增加教学代入和过程案例。调度例切换计数为交错排列 3 次、分组后 1 次。
- chapter_scan/evidence：复用既有 `progress/book-05-llm-training/architecture-audit.md` 第十八章扫描与 `research/llm-training-and-post-training-systems-2026.md` 中 DeepSeek-V4/V4.1、GKD、MiniLLM 和 Model Soups 的版本/证据边界。本轮只访问既有 arXiv 页面确认报告入口，没有扩写报告事实、改技术结论或增加 research 条目。
- code_status：两段 Python 代码均未改，保留“静态审阅，未运行（2026-11-02；Python 标准库）”；没有运行章节代码、测试或模型。
- link/sync_status：标题、章节编号和链接目标不变；整册读者路线复核确认简介摘要仍准确，无需微调；本地静态门禁结果已回填。
- open_risks：V4/V4.1 仍为作者公开披露，路由权重、GRM schema、受控消融、完整成本和复现配置未公开；教学构造用于讲清机制，不补足这些未知信息。章节四部分及末尾报告较密，已给首读顺序；外链未全量在线核验，PDF/GitHub 渲染未检查。

## 四层架构复核与整册阅读路径

- intro_audit：简介保留目标读者前置（Python、反向传播、交叉熵、Transformer 基本数据流；不要求多卡经验）、书册能力、1—14 主线、15—18 专题及第 17→18 先修关系。本轮在“阅读建议”追加一段读者方法说明：核心机制通常从 token、更新、rank 或 checkpoint 等小对象出发，先展示输入、变化和结果，再给术语、规则、必要公式与边界；若章节承诺动手，则由代码/操作步骤兑现。它描述正文实际使用的方式，没有新增学习目标或暗示每节必须有代码。
- toc_audit：简介地图和目录仍列出 18 个章节入口，标题、路径和文件一致。1—14 的阶段顺序从项目与数据进入协议/架构/训练更新，再到效率、稳定性、评估、后训练、多模态、预算和综合诊断；15—18 明确为后训练专题，17 先于依赖一般 OPD 的 18。章数、目录结构、文件名不需调整。
- continuous_read：结合本轮 1—18 逐章完整复读记录，按简介/目录再核对从第 1 章训练地图到第 14 章系统诊断的主线，以及第 11 章后训练基础到第 15 章领域契约、第 16 章 verifier、第 17 章 OPD、第 18 章 V4/V4.1 披露案例的专题路由；复看了主线入口（第 1 章 1.3B/120B/8 卡账本）、综合收束（第 14 章 7B/1T/128 卡）、专题入口（第 15 章财报样本）和 16—18 章新补教学链。没有发现要求重排或拆分章节的路径断层；第 2/3、4/5、5/6、6/7、7/8/9、9/10、10/11、11/12/15—18、12/13/14 和 17/18 的交接均与简介职责一致。
- coverage/depth：章节表及逐章记录表明核心主题保留了必要机制、推导、实现/证据和边界；例子放在本节首次需要它的位置，而不是仅用章首案例代替。例如第 7 章的通信/显存/TP 过程、第 8—10 章的状态与诊断账本、第 11/15—18 章的反馈信号、verifier 与教师状态均有本轮在地案例。未因“连贯”删减必要主题，也没有新增统一字数门槛。
- sync_status：只为简介增加阅读方式说明；该改动不改变章节职责、章名、目标或依赖，目录无需改动。章节标题、编号和本地链接均未在第 16—18 章改动，前面第 1—15 章的目录入口和配套同步沿用各章记录。
- visual/link/formula_status：`git diff --check` 通过。第 16 章反引号围栏 62 个，第 17 章波浪围栏 64 个，第 18 章反引号围栏 28 个，均为偶数；本轮新增行无美元行内公式，无新增链接或标题锚点。此前整册扫描记录的 156 个本地链接仍适用，因为本轮未更换目标或锚点；Mermaid 和 GitHub/PDF 数学渲染没有在本轮执行。
- code_status：本轮未执行代码、测试、训练、benchmark、模型或 PDF。静态未运行的 16.15、17.16 及 18 章两段 Python 示例继续按各章记录标示；其他章节代码状态以逐章记录为准，未因整册阅读伪称已验证。
- open_risks：本地链接、围栏与公式记号按旧门禁和本轮修改范围核对；555 个外链未全量在线访问，章节中的静态代码也未重跑，GitHub/PDF 页面渲染未执行。DeepSeek-V4/V4.1 和 OPD 论文结论仍严格限于作者披露及研究记录列出的模型、任务和配置，相关未知项未以教学例补成事实。

## 未决风险与交接

- 技术结论随模型、论文、框架和 API 版本变化；本轮对修改涉及的新事实做了定向核验，后续修订仍需按规范检查版本与证据边界。
- 本轮已逐章复读并记录必要主题的保留、合并、迁移或删除理由；简介、目录和跨章导航已复核。
- 剩余限制：既有扫描列出的 555 个外链未在本轮全量在线核验；章节静态代码未重跑；GitHub/PDF 公式与页面渲染未检查。此前扫描的 156 个本地链接目标均有记录，本轮没有改动第 16—18 章链接目标或锚点；本地静态门禁结果见上文。
